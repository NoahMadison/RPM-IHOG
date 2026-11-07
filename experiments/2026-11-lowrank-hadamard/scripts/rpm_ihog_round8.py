#!/usr/bin/env python3
r"""
rpm_ihog_round8.py — does the Kronecker core earn its bits?  (controls + low-rank core + sink rule)
====================================================================================================
Same model, full depth, same calib/eval split, same per-sequence bootstrap as rounds 5-7. Everything
new lives here; the round 3-6 modules are imported, never modified.

(a) Controls vs kron13_2b1b_gptq / INT8 factors (3.314 bpw), matched to ±0.05 bpw where the format allows:
    ctrl_gptq_int3_sym   GPTQ INT3, group 64, symmetric, fp16 scale            -> 3.250 bpw (−0.064, NOT matched)
    ctrl_gptq_int3_asym  GPTQ INT3, group 64, fp16 scale + 3-bit zero (AutoGPTQ layout) -> 3.297 bpw (matched)
        no rotation, no core: plain closed-loop GPTQ on the original FFN weights, same per-expert sink-weighted
        Hessians as our pipeline (R6.expert_hessians), static groups. damp chosen per variant from
        --ctrl_damps by CALIBRATION NLL (in-sample; never the eval split); all damps reported.
    kron0_2b1b_gptq      same pipeline, Kronecker rank 0                        -> ~3.08 bpw (NOT matched)
    kron0_2b1b_gg128_gptq  rank 0 + per-group-of-128 residual gains (Cz-weighted LS) to spend the core's bits
                         -> ~3.31 bpw (matched)
(b) lr_2b1b_gptq / lr_2bit_gptq: plain low-rank core U·Vᵀ per expert matrix instead of the Kronecker core,
    init from the truncated SVD of the balanced weight (it is the Kronecker bank with p1=d_out, q1=1, so
    init/train/repack/GPTQ/factor-quantisation are literally the same code). Rank = largest r whose INT8
    factor cost does not exceed Kronecker rank 13 (3B: r=11, 0.230 vs 0.238 bpw).
(c) 2-bit configs (kron13_2bit_gptq_sink, lr_2bit_gptq): in sink-writing layers (token-0 share of the
    teacher FFN output energy > --sink_thresh on the CALIBRATION split) the down matrix is 2b1b instead of
    2bit and its codes are frozen after the init pack (training-time repacks skipped). bpw cost reported.
(d) every row: ΔPPL + CI, median / mean per-sequence ΔNLL, 5 worst sequences, plus top-1, top-5/10 overlap,
    KL, R² hidden/logits vs FP16 (rpm_ihog_eval_agree.agreement).

  CUDA_VISIBLE_DEVICES=1 python rpm_ihog_round8.py --out_dir ./rpm_ihog_round8_3b
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

import rpm_ihog_round3 as R3
import rpm_ihog_round4 as R4
import rpm_ihog_round5 as R5
import rpm_ihog_round6 as R6
import rpm_ihog_eval_agree as AG

FP16 = torch.float16


# ==============================================================================================
# Bank variants (mixins on the round-6 GPTQ banks)
# ==============================================================================================
class LowRankMixin:
    """Kronecker bank with p1=d_out, q1=1, p2=1, q2=d_in  ==  sum_r u_r v_r^T  (plain low rank)."""

    def __init__(self, E, d_out, d_in, rank, res):
        super().__init__(E, d_out, d_in, rank, res)
        self.p1, self.q1, self.p2, self.q2 = d_out, 1, 1, d_in
        if rank > 0:
            self.A = nn.Parameter(torch.zeros(E, rank, d_out, 1, dtype=FP16))
            self.B = nn.Parameter(torch.zeros(E, rank, 1, d_in, dtype=FP16))


class FreezeCodesMixin:
    """Codes are packed once (at init) and never re-quantised during training."""

    @torch.no_grad()
    def repack(self, W):
        if getattr(self, "_packed_once", False):
            return
        self._packed_once = True
        return super().repack(W)


@torch.no_grad()
def fit_grouped_gains_(bank, W):
    """Per-(row, column-group) LS gains for the 2-bit and 1-bit stages, each group solved on its own (2x2),
    columns weighted by diag(Cz) (activation energy) if a Hessian is set.
    (v3 used a joint Cz-weighted solve across groups: badly conditioned, gains exploded -> non-finite loss.)"""
    scale = torch.exp(bank.log_r.float()) * torch.exp(bank.log_c.float())
    K = bank.kron(torch.float32)
    res = W.float() / scale - (K if K is not None else 0.0)
    E, D, N = res.shape
    G = bank.gain.shape[-1]
    gs = N // G
    R2, R1 = bank.rhat().float(), bank.rhat1().float()
    if bank._hess is not None:
        c = torch.exp(bank.log_c.float())                                  # (E, 1, N)
        w = (bank._hess.diagonal(dim1=1, dim2=2).unsqueeze(1) * c * c)     # (E, 1, N) = diag(Cz)
        w = w / w.mean(-1, keepdim=True).clamp_min(1e-12)
    else:
        w = torch.ones(E, 1, N, device=res.device)

    def gsum(x):
        return x.reshape(E, D, G, gs).sum(-1)

    a, b, cc = gsum(w * R2 * R2), gsum(w * R2 * R1), gsum(w * R1 * R1)
    u, v = gsum(w * res * R2), gsum(w * res * R1)
    det = (a * cc - b * b).clamp_min(1e-12)
    bank.gain.copy_(((cc * u - b * v) / det).to(bank.gain.dtype))
    bank.gain1.copy_(((a * v - b * u) / det).to(bank.gain1.dtype))


class GroupGainMixin:
    """2b1b bank whose two stage gains are per (row, group of GS input columns) instead of per row."""
    GS = 128

    def __init__(self, E, d_out, d_in, rank, res):
        super().__init__(E, d_out, d_in, rank, res)
        G = max(1, d_in // self.GS)
        self.gain = nn.Parameter(torch.ones(E, d_out, G, dtype=FP16))
        self.gain1 = nn.Parameter(torch.ones(E, d_out, G, dtype=FP16))

    def _g(self, p, dtype):
        return p.to(dtype).repeat_interleave(self.d_in // p.shape[-1], dim=2)

    def synthesize(self, dtype=torch.float32):
        K = self.kron(dtype)
        W = (K if K is not None else 0.0) + self._g(self.gain, dtype) * self.rhat().to(dtype) \
            + self._g(self.gain1, dtype) * self.rhat1().to(dtype)
        return W * torch.exp(self.log_r.to(dtype)) * torch.exp(self.log_c.to(dtype))

    @torch.no_grad()
    def repack(self, W):
        super().repack(W)                 # codes (GPTQ) + per-row gains, broadcast into the groups
        fit_grouped_gains_(self, W)


_BANKS = {}


def bank_cls(res, core, gs, frozen):
    key = (res, core, gs, frozen)
    if key not in _BANKS:
        base = R6.GBank21 if res == "2b1b" else R6.GBank
        mix = ([FreezeCodesMixin] if frozen else []) + ([GroupGainMixin] if gs else []) + \
              ([LowRankMixin] if core == "lr" else [])
        if gs and res != "2b1b":
            raise ValueError("grouped gains only implemented for 2b1b")
        attrs = {"GS": gs} if gs else {}
        _BANKS[key] = type(f"Bank8_{res}_{core}_g{gs}_{'F' if frozen else ''}", tuple(mix) + (base,), attrs)
    return _BANKS[key]


_BLOCKS = {}


def block_cls8(g, u, d, core, gs=None, freeze_down=False):
    key = (g, u, d, core, gs, freeze_down)
    if key in _BLOCKS:
        return _BLOCKS[key]

    class Block8(R3.Block3):
        def __init__(self, E, d_model, d_ffn, top_k, rank, res=None, whiten=False, expalloc=False, gamma=0.5):
            super().__init__(E, d_model, d_ffn, top_k, rank, "1bit", whiten, expalloc, gamma)
            self.gate = bank_cls(g, core, gs, False)(E, d_ffn, d_model, rank, g)
            self.up = bank_cls(u, core, gs, False)(E, d_ffn, d_model, rank, u)
            self.down = bank_cls(d, core, gs, freeze_down)(E, d_model, d_ffn, rank, d)

        @torch.no_grad()
        def init_from_teacher(self, router_w, w_in, w_out, X, Y, dev):
            if R6.CTX["hess"] is not None:
                Cin, Cmid = R6.CTX["hess"]
                self.gate._hess, self.up._hess, self.down._hess = Cin, Cin, Cmid
            return super().init_from_teacher(router_w, w_in, w_out, X, Y, dev)

    Block8.__name__ = Block8.__qualname__ = f"Block8_{g}_{u}_{d}_{core}_g{gs}_{'F' if freeze_down else ''}"
    _BLOCKS[key] = Block8
    return Block8


def lr_rank_matched(d_model, d_ffn, kron_rank=13, fb=8, p1=32, q1=32):
    """Largest low-rank r whose INT{fb} factor bits (incl. one fp16 scale per factor) <= the Kronecker core's."""
    kron_terms = p1 * q1 + (d_ffn // p1) * (d_model // q1)       # same for gate/up (d_ffn x d_model) and down
    kron_bits = kron_rank * (fb * kron_terms + 32)
    return int(kron_bits // (fb * (d_model + d_ffn) + 32))


def make_configs(lr_rank):
    def c(tag, g, u, d, core, rank, gs=None, sink_rule=False):
        return {"tag": tag, "rank": rank, "res": "mix", "core": core, "gs": gs, "sink_rule": sink_rule, "gptq": True,
                "residuals": {"gate": g, "up": u, "down": d}, "block_cls": block_cls8(g, u, d, core, gs)}
    t = "2b1b"
    return {
        "kron13_2b1b_gptq": c("kron13_2b1b_gptq", t, t, t, "kron", 13),
        "lr_2b1b_gptq": c("lr_2b1b_gptq", t, t, t, "lr", lr_rank),
        "kron0_2b1b_gg128_gptq": c("kron0_2b1b_gg128_gptq", t, t, t, "none", 0, gs=128),
        "kron0_2b1b_gptq": c("kron0_2b1b_gptq", t, t, t, "none", 0),
        "kron13_2bit_gptq_sink": c("kron13_2bit_gptq_sink", "2bit", "2bit", "2bit", "kron", 13, sink_rule=True),
        "lr_2bit_gptq": c("lr_2bit_gptq", "2bit", "2bit", "2bit", "lr", lr_rank, sink_rule=True),
    }


# ==============================================================================================
# GPTQ INT3 control (no rotation, no core)
# ==============================================================================================
@torch.no_grad()
def gptq_int3_(P, H, damp, sym, g=64):
    """In-place GPTQ of an (E, R, C) expert weight with input Hessian H (E, C, C), static groups of g columns."""
    X = P.data.float()
    E, R, C = X.shape
    assert H.shape == (E, C, C), (H.shape, X.shape)
    Xg = X.reshape(E, R, C // g, g)
    # scales are stored fp16; floor at the smallest fp16 subnormal (2^-24) so near-constant groups (the 3B has
    # groups with range ~1e-32 in L0-3) don't underflow to 0 and produce 0/0 = NaN.
    if sym:
        s = (Xg.abs().amax(-1).clamp_min(1e-8) / 3).half().float().clamp_min(2.0 ** -24)
        z = None
    else:
        mn, mx = Xg.amin(-1), Xg.amax(-1)
        s = ((mx - mn).clamp_min(1e-8) / 7).half().float().clamp_min(2.0 ** -24)
        z = torch.round(-mn / s).clamp(0, 7)
    Q = torch.empty_like(X)

    def quant(w, j):
        gi = j // g
        sc = s[:, :, gi]
        if sym:
            q = torch.round(w / sc).clamp(-4, 3) * sc
        else:
            zz = z[:, :, gi]
            q = ((torch.round(w / sc) + zz).clamp(0, 7) - zz) * sc
        Q[:, :, j] = q
        return q

    R6.gptq(X, H, quant, damp, 128)
    P.data.copy_(Q.to(P.dtype))


def run_gptq_ctrl(model, orig, layers, calib, Yteach, dev, sym, damp):
    for l in layers:
        Xci, _, _ = R3.capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
        rw, wi, wo, tk = R4.extract(orig[l], dev)
        Cin, Cmid, _ = R6.expert_hessians(rw, wi, tk, Xci, Yteach[l].to(dev))
        p_in, p_out = R4.ffn_params(orig[l])
        assert p_in.shape == wi.shape and p_out.shape == wo.shape, "unexpected FFN layout for the INT3 control"
        gptq_int3_(p_in, Cin, damp, sym)
        gptq_int3_(p_out, Cmid, damp, sym)
        if not (torch.isfinite(p_in).all() and torch.isfinite(p_out).all()):
            raise RuntimeError(f"INT3 control produced non-finite weights at layer {l} (sym={sym}, damp={damp})")
        del Xci, Cin, Cmid, wi, wo
        torch.cuda.empty_cache()


# ==============================================================================================
# Chain with per-layer sink rule
# ==============================================================================================
def run_chain8(name, spec, model, orig, layers, calib, evalt, dev, args, sinks):
    replaced, per = [], {}
    g, u, d = spec["residuals"].values()
    for l in layers:
        t = R4.teacher_for(model, orig, replaced, l, calib, evalt, dev)
        if replaced:
            Xci, _, _ = R3.capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
            Xei, _, _ = R3.capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
        else:
            Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
        Cin, Cmid, ntok = R6.expert_hessians(t["router_w"], t["w_in"], t["top_k"], Xci, t["Y_calib_teacher"])
        sk = bool(spec["sink_rule"] and l in sinks)
        sp = {**spec, "block_cls": block_cls8(g, u, "2b1b" if sk else d, spec["core"], spec["gs"], freeze_down=sk)}
        R6.CTX["hess"] = (Cin, Cmid)
        try:
            blk, m = R5.train_layer5(name, sp, t, Xci, Xei, dev, args, os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"))
        finally:
            R6.CTX["hess"] = None
        m["hess_tokens_min"], m["hess_tokens_median"] = min(ntok), sorted(ntok)[len(ntok) // 2]
        m["sink_fix"] = sk
        R4.set_ffn(model, l, blk)
        replaced.append(l)
        per[l] = m
        print(f"  L{l:2d} bpw={m['bpw']:.3f}{' SINKFIX' if sk else ''} open={m['open_r2']:6.1f} (no tok0 "
              f"{m['open_r2_nopos0']:5.1f}) closed={m['closed_r2']:6.1f} wR2 g/u/d={m['wR2_gate']:.0f}/"
              f"{m['wR2_up']:.0f}/{m['wR2_down']:.0f} ({m['train_sec']}s) mem {torch.cuda.memory_allocated() / 2**30:.1f}G", flush=True)
        del t, Xci, Xei, Cin, Cmid
        torch.cuda.empty_cache()
    return per


# ==============================================================================================
# Scoring: ΔPPL + per-sequence + agreement
# ==============================================================================================
REF = {}


def score(model, evalt, dev, tok, label, extra):
    nll = R4.seq_nll(model, evalt, dev)
    lg, h = AG.forward_all(model, evalt, dev)
    ag = AG.agreement(lg, h, REF["lg"], REF["h"], dev)
    del lg, h
    dn = (nll - REF["nll"]).float()
    worst = torch.argsort(dn, descending=True)[:5].tolist()
    row = {"name": label, **extra, **R4.ppl_stats(nll, REF["nll"]),
           "median_dnll": round(float(dn.median()), 4), "mean_dnll": round(float(dn.mean()), 4),
           "nll_seq": [round(float(x), 5) for x in nll.tolist()],
           "worst5": [{"seq": i, "dnll": round(float(dn[i]), 4),
                       "text": tok.decode(evalt[i, :24].tolist()).replace("\n", " ")[:80]} for i in worst],
           **ag}
    b = row.get("bpw")
    print(f"  {label:<34} bpw={b if b is None else f'{b:.3f}'}  PPL {row['ppl']} (Δ {row['dppl']:+} CI {row['ci95']})  "
          f"medΔNLL {row['median_dnll']:+.4f}  top1 {ag['top1']}% top5 {ag['top5_overlap']}%  KL {ag['kl_per_tok']}  "
          f"R²h {ag['r2_hidden']}  worst {[w['seq'] for w in row['worst5']]}", flush=True)
    return row


# ==============================================================================================
# Driver
# ==============================================================================================
def run_round8(model, tok, calib, evalt, args, dev):
    os.makedirs(args.out_dir, exist_ok=True)
    nL = len(model.model.layers)
    layers = args.layers or list(range(nL))
    orig = {l: R4.get_ffn(model, l) for l in range(nL)}
    _, wi0, _, _ = R4.extract(orig[0], dev)
    d_ffn, d_model = wi0.shape[1] // 2, wi0.shape[2]
    lr_rank = lr_rank_matched(d_model, d_ffn)
    CONF = make_configs(lr_rank)
    R6.GPTQ["damp"] = args.damp

    REF["nll"] = R4.seq_nll(model, evalt, dev)
    REF["lg"], REF["h"] = AG.forward_all(model, evalt, dev)
    res = {"model": args.model_id, "n_layers": nL, "layers": layers, "fp16_ppl": round(float(REF["nll"].mean().exp()), 3),
           "lr_rank": lr_rank, "args": vars(args), "sinks": {}, "rtn": [], "controls": [], "configs": {},
           "fp16_nll_seq": [round(float(x), 5) for x in REF["nll"].tolist()],
           "spectral": {}, "created": time.strftime("%Y-%m-%d %H:%M")}
    path = os.path.join(args.out_dir, "results.json")

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)

    print(f"[*] {args.model_id}: {nL} layers; FP16 PPL {res['fp16_ppl']}; low-rank core rank {lr_rank}", flush=True)

    # --- clean teacher outputs on calib (sink detection + INT3-control Hessian weights) ---
    Yteach, frac = {}, {}
    for l in range(nL):
        _, Y, _ = R3.capture_real_moe_io(model, orig[l], calib, dev)
        e = Y.float().pow(2).sum(1)
        T = Y.shape[0] // calib.shape[0]
        pos0 = (torch.arange(Y.shape[0], device=Y.device) % T) == 0
        frac[l] = round(float(e[pos0].sum() / e.sum()), 4)
        Yteach[l] = Y.cpu()
        del Y, e
    sinks = [l for l in layers if frac[l] > args.sink_thresh]
    res["sinks"] = {"thresh": args.sink_thresh, "tok0_energy_calib": frac, "sink_layers": sinks}
    print(f"[*] sink-writing layers (calib tok0 energy > {args.sink_thresh}): {sinks}", flush=True)
    save()

    # --- RTN references ---
    if args.rtn_bits:
        print("\n=== RTN reference (all FFN weights, group 64) ===", flush=True)
        backup = [p.data.cpu().clone() for l in range(nL) for p in R4.ffn_params(orig[l])]
        for bits in args.rtn_bits:
            for l in range(nL):
                for p in R4.ffn_params(orig[l]):
                    R4.rtn_(p, bits)
            res["rtn"].append(score(model, evalt, dev, tok, f"RTN INT{bits} g64", {"bits": bits, "bpw": bits + 0.25}))
            with torch.no_grad():
                for p, dd in zip((p for l in range(nL) for p in R4.ffn_params(orig[l])), backup):
                    p.data.copy_(dd.to(p.device))
        del backup
        save()

    # --- (a) GPTQ INT3 controls ---
    if args.controls:
        print("\n=== (a) GPTQ INT3 g64 controls (no rotation, no core) ===", flush=True)
        backup = [p.data.cpu().clone() for l in range(nL) for p in R4.ffn_params(orig[l])]
        for var in args.controls:
            sym = var == "sym"
            bpw_ = 3 + 16 / 64 if sym else 3 + (16 + 3) / 64
            rows = []
            for damp in args.ctrl_damps:
                t0 = time.perf_counter()
                run_gptq_ctrl(model, orig, layers, calib, Yteach, dev, sym, damp)
                nll_cal = R4.seq_nll(model, calib, dev)
                r = score(model, evalt, dev, tok, f"ctrl_gptq_int3_{var} d{damp}",
                          {"variant": var, "damp": damp, "bpw": round(bpw_, 4),
                           "calib_nll": round(float(nll_cal.mean()), 5), "sec": round(time.perf_counter() - t0, 1)})
                rows.append(r)
                with torch.no_grad():
                    for p, dd in zip((p for l in range(nL) for p in R4.ffn_params(orig[l])), backup):
                        p.data.copy_(dd.to(p.device))
            best = min(rows, key=lambda r: r["calib_nll"])
            for r in rows:
                r["selected_on_calib"] = r is best
            print(f"  -> {var}: damp {best['damp']} selected on calib NLL", flush=True)
            res["controls"] += rows
            save()
        del backup

    # --- (a)(b)(c) pipeline configs ---
    for c in args.configs:
        spec = CONF[c]
        print(f"\n=== {c}  core={spec['core']} rank={spec['rank']} gs={spec['gs']} residuals="
              f"{'/'.join(spec['residuals'].values())} sink_rule={spec['sink_rule']} on {len(layers)} layers ===", flush=True)
        per = run_chain8(c, spec, model, orig, layers, calib, evalt, dev, args, sinks)
        blocks = [R4.get_ffn(model, l) for l in layers]
        rows = []
        for fb in (args.factor_bits if spec["rank"] > 0 else [16]):
            saved = R4.quantize_factors_(blocks, fb) if fb < 16 else []
            bd = {k: sum(R5.bpw_breakdown(b, fb)[k] for b in blocks) / len(blocks) for k in R5.bpw_breakdown(blocks[0], fb)}
            label = f"{c} f{'16' if fb == 16 else f'INT{fb}'}" if spec["rank"] > 0 else c
            rows.append(score(model, evalt, dev, tok, label, {"fb": fb, "bpw": round(sum(bd.values()), 4), "breakdown": bd}))
            R4.restore_(saved)
        sink_cost = None
        if spec["sink_rule"]:
            plain = [R5.bpw_breakdown(b, 8) for b, l in zip(blocks, layers) if l not in sinks]
            if plain:
                b_plain = sum(plain[0].values())
                b_all = sum(sum(R5.bpw_breakdown(b, 8).values()) for b in blocks) / len(blocks)
                sink_cost = round(b_all - b_plain, 4)
                print(f"  -> sink rule bpw cost (INT8 factors): +{sink_cost} ({len([l for l in layers if l in sinks])} layers)",
                      flush=True)
        R4.restore_model(model, orig, layers)
        for b in blocks:                      # drop decoded-residual caches, then the blocks themselves
            for bk in b.banks():
                bk._rhat = None
                if hasattr(bk, "_rhat1"):
                    bk._rhat1 = None
        del blocks, saved
        torch.cuda.empty_cache()
        res["configs"][c] ={k: v for k, v in spec.items() if k != "block_cls"}
        res["configs"][c].update({"rows": rows, "sink_cost_bpw": sink_cost, "layers": {str(l): m for l, m in per.items()}})
        save()

    # --- matched-bpw check vs kron13_2b1b_gptq INT8 ---
    ref = next((r for r in res["configs"].get("kron13_2b1b_gptq", {}).get("rows", []) if r["fb"] == 8), None)
    if ref:
        print(f"\n=== bpw match vs kron13_2b1b_gptq INT8 ({ref['bpw']:.3f}) ===", flush=True)
        allrows = res["controls"] + [r for v in res["configs"].values() for r in v["rows"]]
        a_ref = torch.tensor(ref["nll_seq"])
        g = torch.Generator().manual_seed(0)
        idx = torch.randint(0, a_ref.numel(), (4000, a_ref.numel()), generator=g)
        for r in allrows:
            r["dbpw_vs_ref"] = round(r["bpw"] - ref["bpw"], 4)
            r["matched_005"] = abs(r["dbpw_vs_ref"]) <= 0.05
            # paired (same eval seqs) bootstrap of PPL(row) - PPL(ref); > 0 means the row is worse than the reference
            a = torch.tensor(r["nll_seq"])
            d = a[idx].mean(1).exp() - a_ref[idx].mean(1).exp()
            lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975])).tolist()
            r["paired_vs_ref"] = {"dppl": round(float(a.mean().exp() - a_ref.mean().exp()), 3), "ci95": [round(lo, 3), round(hi, 3)]}
        for r in allrows:
            pv = r["paired_vs_ref"]
            print(f"  {r['name']:<34} bpw {r['bpw']:.3f} ({r['dbpw_vs_ref']:+.3f}) {'MATCHED' if r['matched_005'] else '       '}"
                  f"  Δ {r['dppl']:+} CI {r['ci95']}  paired vs ref {pv['dppl']:+} CI {pv['ci95']}", flush=True)
        save()

    # --- spectral on the 2b1b cores (uniform per-layer classes) ---
    import rpm_ihog_spectral as SP
    spec_cfgs = [c for c in args.configs if not CONF[c]["sink_rule"]]
    for l in args.spectral_layers:
        if l >= nL or not spec_cfgs:
            continue
        print(f"\n=== spectral analysis, layer {l} ({spec_cfgs}) ===", flush=True)
        try:
            res["spectral"][str(l)] = SP.analyze(model, l, args.out_dir, spec_cfgs,
                                                 os.path.join(args.out_dir, f"spectral_L{l}"), evalt, dev, cfg_table=CONF)
        except SystemExit as e:
            print(f"[!] spectral L{l} skipped: {e}")
        save()
    print("\n[round 8] done", flush=True)
    return res


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
    ap.add_argument("--configs", nargs="*", default=list(make_configs(11)))
    ap.add_argument("--controls", nargs="*", default=["asym", "sym"], choices=["asym", "sym"])
    ap.add_argument("--ctrl_damps", type=float, nargs="+", default=[0.01, 0.1, 1.0])
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: ALL layers")
    ap.add_argument("--rtn_bits", type=int, nargs="*", default=[4, 3])
    ap.add_argument("--factor_bits", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--spectral_layers", type=int, nargs="*", default=[5, 12])
    ap.add_argument("--sink_thresh", type=float, default=0.5)
    ap.add_argument("--damp", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--batch_tokens", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", required=True)
    return ap


def main():
    args = build_parser().parse_args()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"{args.out_dir} exists and is not empty; pick a fresh --out_dir")
    print("[round 8] controls (GPTQ INT3, rank 0), low-rank core at equal bits, sink rule for 2-bit, per-seq ΔNLL.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = R4.V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=FP16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    run_round8(model, tok, calib, evalt, args, dev)


if __name__ == "__main__":
    main()
