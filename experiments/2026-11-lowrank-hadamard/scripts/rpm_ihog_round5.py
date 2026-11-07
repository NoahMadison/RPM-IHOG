#!/usr/bin/env python3
r"""
rpm_ihog_round5.py — fixes from the 3B post-mortem, full-depth, with a full report
=================================================================================
What changed vs round 4 (each traced to a measured failure):

  1. Sink-proof objective. Distillation loss and the output-gain fit are now PER-TOKEN RELATIVE:
         L = sum_t w_t |y_t - yhat_t|^2 / sum_t w_t |y_t|^2,   w_t = 1 / (|y_t|^2 + median_t |y_t|^2)
     On the 3B, layers 3/5 put ~100% of their output energy on token 0; plain MSE let that one token
     set a global gain of 14x and wreck every other token (R^2 -2345%) and everything downstream.
     Now each token counts once, the sink token included (it still has to be reproduced).
  2. Matrix-level bits. The gate is the fragile matrix (single-swap R^2 35-54% vs 68-78% for up/down,
     because SiLU turns small gate errors into neurons switching on/off). New config gives the gate a
     2-bit residual and keeps up/down at 1-bit.
  3. More diagnostics per layer, all on clean held-out inputs: one-shot R^2 on clean AND drifted
     inputs, R^2 without token 0, weight-space R^2 per matrix, single-matrix swap R^2, share of output
     energy on token 0, per-token relative-error quantiles, training curves.

Configs (Kronecker rank 13 everywhere; factors scored at fp16 / INT8 / INT4):
  kron13_1bit   1-bit residual on gate/up/down
  gate2_1bit    2-bit gate, 1-bit up/down
  kron13_2bit   2-bit residual everywhere
  kron13_2b1b   2-bit + 1-bit stacked residual (the v8 recipe)
plus RTN INT{8,4,3,2} g64 on all FFN weights as the reference frontier.

Then it runs the spectral analysis on --spectral_layers and writes the report (rpm_ihog_report.py):
figures as PNG + one multi-page PDF + REPORT.md. Regenerate the report alone with
  python rpm_ihog_report.py <out_dir>/results.json

  CUDA_VISIBLE_DEVICES=1 python rpm_ihog_round5.py --model_id ibm-granite/granite-3.1-3b-a800m-base
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

import rpm_ihog_round3 as R3
import rpm_ihog_round4 as R4

# ------------------------------------------------------------------------------
# Configs: per-matrix residual types
# ------------------------------------------------------------------------------
_MIX_CACHE = {}


def mix_cls(gate, up, down):
    key = (gate, up, down)
    if key in _MIX_CACHE:
        return _MIX_CACHE[key]

    class BlockMix(R3.Block3):
        def __init__(self, E, d_model, d_ffn, top_k, rank, res=None, whiten=False, expalloc=False, gamma=0.5):
            super().__init__(E, d_model, d_ffn, top_k, rank, "1bit", whiten, expalloc, gamma)

            def mk(r, do, di):
                return R4.Bank21(E, do, di, rank) if r == "2b1b" else R3.Bank(E, do, di, rank, r)

            self.gate = mk(gate, d_ffn, d_model)
            self.up = mk(up, d_ffn, d_model)
            self.down = mk(down, d_model, d_ffn)

    BlockMix.__name__ = BlockMix.__qualname__ = f"BlockMix_{gate}_{up}_{down}"
    BlockMix.residuals = {"gate": gate, "up": up, "down": down}
    _MIX_CACHE[key] = BlockMix
    return BlockMix


def _cfg(tag, g, u, d, rank=13):
    return {"tag": tag, "rank": rank, "res": "mix", "block_cls": mix_cls(g, u, d), "residuals": {"gate": g, "up": u, "down": d}}


CONFIGS5 = {
    "kron13_1bit": _cfg("kron13_1bit", "1bit", "1bit", "1bit"),
    "gate2_1bit": _cfg("gate2_1bit", "2bit", "1bit", "1bit"),
    "kron13_2bit": _cfg("kron13_2bit", "2bit", "2bit", "2bit"),
    "kron13_2b1b": _cfg("kron13_2b1b", "2b1b", "2b1b", "2b1b"),
}


# ------------------------------------------------------------------------------
# Objective
# ------------------------------------------------------------------------------
def rel_mse(yp, y):
    e = y.pow(2).sum(1)
    w = 1.0 / (e + e.median() + 1e-12)
    return (w * (yp - y).pow(2).sum(1)).sum() / (w * e).sum()


@torch.no_grad()
def refit_gain_(blk, X, Y, n=4096):
    x, y = X[:n].float(), Y[:n].float()
    blk.out_gain.fill_(1.0)
    yp = blk.forward_moe(x)
    e = y.pow(2).sum(1)
    w = 1.0 / (e + e.median() + 1e-12)
    a = float((w * (yp * y).sum(1)).sum() / ((w * yp.pow(2).sum(1)).sum() + 1e-12))
    blk.out_gain.fill_(min(max(a, 0.05), 20.0))


# ------------------------------------------------------------------------------
# Diagnostics (clean held-out inputs)
# ------------------------------------------------------------------------------
def _r2(P, T):
    return 100.0 * (1.0 - float((P - T).pow(2).mean() / T.var()))


@torch.no_grad()
def diagnose(blk, t, seq_len):
    X, Y = t["X_eval_clean"].float(), t["Y_eval_teacher"].float()
    keep = (torch.arange(X.shape[0], device=X.device) % seq_len) != 0
    f = t["w_in"].shape[1] // 2
    exact = (t["w_in"][:, :f].float(), t["w_in"][:, f:].float(), t["w_out"].float())
    comp = tuple(W.float() for W in blk.synth_all(torch.float32))

    def run(Ws, gain):
        g0 = blk.out_gain.data.clone()
        blk.out_gain.data.fill_(gain)
        P = torch.cat([blk.forward_moe(X[i : i + 2048], Ws) for i in range(0, X.shape[0], 2048)])
        blk.out_gain.data.copy_(g0)
        return P

    P = run(comp, float(blk.out_gain))
    out = {"open_r2_nopos0": _r2(P[keep], Y[keep]), "out_gain": float(blk.out_gain)}
    for k, name in enumerate(("gate", "up", "down")):
        Ws = list(exact)
        Ws[k] = comp[k]
        Pk = run(tuple(Ws), 1.0)
        out[f"swap_{name}"] = _r2(Pk[keep], Y[keep])
        out[f"wR2_{name}"] = 100.0 * (1.0 - float((comp[k] - exact[k]).pow(2).sum() / exact[k].pow(2).sum()))
    e = Y.pow(2).sum(1)
    rel = ((P - Y).pow(2).sum(1) / (e + 1e-12)).sqrt()
    q = torch.quantile(rel[keep], torch.tensor([0.5, 0.9, 0.99], device=rel.device)).tolist()
    out.update({"y_pos0": float(e[~keep].sum() / e.sum()), "relerr_p50": q[0], "relerr_p90": q[1],
                "relerr_p99": q[2], "relerr_sink": float(rel[~keep].mean())})
    return out


def bpw_breakdown(blk, fb=16):
    n = blk.E * 3 * blk.gate.d_out * blk.gate.d_in
    cats = {"residual codes": 0.0, "Kronecker factors": 0.0, "scales & gains": 0.0}
    for k, v in blk.state_dict().items():
        name = k.split(".")[-1]
        if k == "router_w":
            continue
        if name in ("A", "B"):
            bank = getattr(blk, k.split(".")[0])
            active = bank.rank_mask.sum().item()
            cats["Kronecker factors"] += active * (fb * v[0, 0].numel() + (16 if fb < 16 else 0))
        elif name in ("packed", "packed1"):
            cats["residual codes"] += 8 * v.numel()
        elif name == "rank_mask":
            cats["scales & gains"] += 8 * v.shape[0]
        else:
            cats["scales & gains"] += 8 * v.numel() * v.element_size()
    return {k: v / n for k, v in cats.items()}


# ------------------------------------------------------------------------------
# Per-layer training
# ------------------------------------------------------------------------------
def train_layer5(name, spec, t, X_cal_in, X_ev_in, dev, args, ckpt):
    E, two_f, d = t["w_in"].shape
    cls = spec["block_cls"]

    def mk():
        b = cls(E, d, two_f // 2, t["top_k"], spec["rank"], spec["res"]).to(dev)
        b._return_tuple = t["returns_tuple"]
        return b

    blk = mk()
    blk.init_from_teacher(t["router_w"], t["w_in"], t["w_out"], t["X_calib_clean"], t["Y_calib_teacher"], dev)
    refit_gain_(blk, t["X_calib_clean"], t["Y_calib_teacher"])
    oneshot_closed = R3.eval_block(blk, X_ev_in, t["Y_eval_teacher"], dev)["r2_pct"]
    oneshot_open = R3.eval_block(blk, t["X_eval_clean"], t["Y_eval_teacher"], dev)["r2_pct"]

    wg, wu = t["w_in"].chunk(2, dim=1)
    teach = (wg, wu, t["w_out"])
    teach_pow = [T.pow(2).mean() for T in teach]
    blk.train()
    params = [p for p in blk.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    scale_ids = {id(b.log_r) for b in blk.banks()} | {id(b.log_c) for b in blk.banks()}
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if id(p) not in scale_ids], "lr": args.lr},
         {"params": [p for p in params if id(p) in scale_ids], "lr": args.lr_scale}],
        betas=(0.9, 0.95), weight_decay=0.0)
    steps = args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    X_clean, Y = t["X_calib_clean"], t["Y_calib_teacher"]
    N = X_cal_in.shape[0]
    var = Y.float().var().item() + 1e-12
    gen = torch.Generator(device=dev).manual_seed(getattr(args, "seed", 42))
    repack_at = {int(0.30 * steps), int(0.60 * steps)}
    hist, t0 = [], time.perf_counter()

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (args.batch_tokens,), generator=gen, device=dev)
        x, xc, y = X_cal_in[idx].float(), X_clean[idx].float(), Y[idx].float()
        Ws = blk.synth_all(torch.float32)
        yp = blk.forward_moe(x, Ws, router_x=xc if step < int(0.65 * steps) else None)
        rl = rel_mse(yp, y)
        wl = sum(((W - T).pow(2).mean() / P) for W, T, P in zip(Ws, teach, teach_pow))
        loss = rl + args.anchor * wl
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{name}] non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step == 1 or step % 10 == 0 or step == steps:
            hist.append([step, float(rl), float(F.mse_loss(yp, y) / var)])
        if step in repack_at:
            for b, T in zip(blk.banks(), teach):
                b.repack(T)
                for g in (getattr(b, "gain", None), getattr(b, "gain1", None)):
                    if g is not None:
                        opt.state.pop(g, None)

    with torch.no_grad():
        for p in params:
            p.data = p.data.half()
    dt = time.perf_counter() - t0

    torch.save(blk.state_dict(), ckpt)
    re = mk()
    re.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True), strict=True)
    re.eval()
    m = {
        "bpw": round(R4.bpw(re), 4),
        "oneshot_closed": oneshot_closed, "oneshot_open": oneshot_open,
        "closed_r2": R3.eval_block(re, X_ev_in, t["Y_eval_teacher"], dev)["r2_pct"],
        "open_r2": R3.eval_block(re, t["X_eval_clean"], t["Y_eval_teacher"], dev)["r2_pct"],
        "train_sec": round(dt, 1), "hist": hist,
    }
    m.update(diagnose(re, t, args.seq_len))
    return re, m


def run_chain5(name, spec, model, orig, layers, calib, evalt, dev, args):
    replaced, per = [], {}
    for l in layers:
        t = R4.teacher_for(model, orig, replaced, l, calib, evalt, dev)
        if replaced:
            Xci, _, _ = R3.capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
            Xei, _, _ = R3.capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
        else:
            Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
        blk, m = train_layer5(name, spec, t, Xci, Xei, dev, args, os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"))
        R4.set_ffn(model, l, blk)
        replaced.append(l)
        per[l] = m
        print(f"  L{l:2d} bpw={m['bpw']:.3f} 1shot open/closed={m['oneshot_open']:6.1f}/{m['oneshot_closed']:6.1f}  "
              f"open={m['open_r2']:6.1f} (no tok0 {m['open_r2_nopos0']:5.1f})  closed={m['closed_r2']:6.1f}  "
              f"gain={m['out_gain']:.2f}  tok0={m['y_pos0']:.0%}  ({m['train_sec']}s)", flush=True)
        del t, Xci, Xei
        torch.cuda.empty_cache()
    return per


# ------------------------------------------------------------------------------
# Driver (model-agnostic so it can be smoke-tested without downloading anything)
# ------------------------------------------------------------------------------
def run_round5(model, calib, evalt, args, dev):
    os.makedirs(args.out_dir, exist_ok=True)
    nL = len(model.model.layers)
    layers = args.layers or list(range(nL))
    orig = {l: R4.get_ffn(model, l) for l in range(nL)}
    kind = "dense" if R4.is_dense(orig[0]) else "MoE"
    nll_fp16 = R4.seq_nll(model, evalt, dev)
    res = {"model": args.model_id, "kind": kind, "n_layers": nL, "layers": layers,
           "fp16_ppl": round(float(nll_fp16.mean().exp()), 3),
           "args": {k: v for k, v in vars(args).items()}, "rtn": [], "configs": {},
           "spectral": {}, "created": time.strftime("%Y-%m-%d %H:%M")}
    path = os.path.join(args.out_dir, "results.json")

    def save():
        with open(path, "w") as f:
            json.dump(res, f, indent=1, default=str)

    print(f"[*] {args.model_id}: {kind}, {nL} layers; FP16 PPL {res['fp16_ppl']}")
    if args.rtn_bits:
        print("\n=== RTN reference (all FFN weights, group 64) ===")
        backup = [p.data.cpu().clone() for l in range(nL) for p in R4.ffn_params(orig[l])]
        for bits in args.rtn_bits:
            for l in range(nL):
                for p in R4.ffn_params(orig[l]):
                    R4.rtn_(p, bits)
            r = {"name": f"RTN INT{bits}", "bits": bits, "bpw": bits + 0.25,
                 **R4.ppl_stats(R4.seq_nll(model, evalt, dev), nll_fp16)}
            res["rtn"].append(r)
            print(f"  {r['name']:<9} bpw={r['bpw']:<5} PPL {r['ppl']} (Δ {r['dppl']:+} CI {r['ci95']})", flush=True)
            with torch.no_grad():
                for p, dd in zip((p for l in range(nL) for p in R4.ffn_params(orig[l])), backup):
                    p.data.copy_(dd.to(p.device))
        del backup
        save()

    for c in args.configs:
        spec = CONFIGS5[c]
        print(f"\n=== {c}  (gate/up/down residual: {'/'.join(spec['residuals'].values())}) on {len(layers)} layers ===")
        per = run_chain5(c, spec, model, orig, layers, calib, evalt, dev, args)
        blocks = [R4.get_ffn(model, l) for l in layers]
        rows = []
        for fb in args.factor_bits:
            saved = R4.quantize_factors_(blocks, fb) if fb < 16 else []
            nll = R4.seq_nll(model, evalt, dev)
            bd = {k: sum(bpw_breakdown(b, fb)[k] for b in blocks) / len(blocks) for k in bpw_breakdown(blocks[0], fb)}
            r = {"fb": fb, "bpw": round(sum(bd.values()), 4), "breakdown": bd, **R4.ppl_stats(nll, nll_fp16)}
            rows.append(r)
            print(f"  -> factors {'fp16' if fb == 16 else f'INT{fb}'}: bpw={r['bpw']:.3f}  PPL {r['ppl']} "
                  f"(Δ {r['dppl']:+} CI {r['ci95']})", flush=True)
            R4.restore_(saved)
        R4.restore_model(model, orig, layers)
        torch.cuda.empty_cache()
        res["configs"][c] = {"residuals": spec["residuals"], "rank": spec["rank"], "rows": rows,
                             "layers": {str(l): m for l, m in per.items()}}
        save()

    for l in args.spectral_layers:
        if l >= nL:
            continue
        print(f"\n=== spectral analysis, layer {l} ===")
        import rpm_ihog_spectral as SP
        try:
            s = SP.analyze(model, l, args.out_dir, list(args.configs), os.path.join(args.out_dir, f"spectral_L{l}"),
                           evalt, dev, cfg_table=CONFIGS5)
            res["spectral"][str(l)] = s
        except SystemExit as e:
            print(f"[!] spectral L{l} skipped: {e}")
        save()

    import rpm_ihog_report as RP
    RP.make_report(path)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS5), choices=list(CONFIGS5))
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: ALL layers")
    ap.add_argument("--rtn_bits", type=int, nargs="*", default=[8, 4, 3, 2])
    ap.add_argument("--factor_bits", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--spectral_layers", type=int, nargs="*", default=[3, 12])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--batch_tokens", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    args.out_dir = args.out_dir or f"./rpm_ihog_round5_{args.model_id.split('/')[-1]}"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = R4.V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    run_round5(model, calib, evalt, args, dev)


if __name__ == "__main__":
    main()
