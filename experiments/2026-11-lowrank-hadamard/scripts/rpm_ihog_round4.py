#!/usr/bin/env python3
r"""
rpm_ihog_round4.py — "is it usable?"
====================================
Full-depth compression (EVERY FFN/MoE layer, not 5 of 24) on any Granite-family model,
MoE or dense, scored against round-to-nearest INT-k group-64 baselines at MATCHED bpw.

Experiments (--exps):
  rtn       RTN INT{8,4,3,2} g64 on all FFN weights            -> 8.25 / 4.25 / 3.25 / 2.25 bpw
  frontier  full-depth closed-loop RPM-IHOG, three configs:
               kron13_1bit   Kronecker + 1-bit Hadamard residual
               kron13_2bit   Kronecker + 2-bit Hadamard residual
               kron13_2b1b   Kronecker + 2-bit + 1-bit residual (the v8 recipe)
            each scored with Kronecker factors at fp16, INT8 and INT4 (post-hoc, per-matrix scale).
            With INT4 factors: 2bit ~2.3 bpw (vs RTN INT2 2.25), 2b1b ~3.25 bpw (vs RTN INT3 3.25).
  noise     kron13_1bit on 5 layers, 3 seeds -> how big is run-to-run noise in dPPL?
  fullalloc (opt-in) full-depth per-layer allocation over {1bit, 2bit, 2b1b}, Lagrangian search.

  --teacher_quant 8|4   quantize the model's FFN weights first (RTN g64) and compress THAT model.

Every PPL comes with a 95% bootstrap CI over eval sequences. Everything stays on the GPU except a
one-time CPU backup of the FFN weights for the RTN sweep (weights, not training data).

Models (same family, same tokenizer/data):
  ibm-granite/granite-3.1-1b-a400m-base   MoE   (24 layers, 32 experts)
  ibm-granite/granite-3.1-3b-a800m-base   MoE   (bigger; hidden 1536 -> block-Hadamard)
  ibm-granite/granite-3.1-2b-base         dense (~same total size as the 3b MoE)
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import rpm_ihog_15bpw_hadamard_distill as V1
import rpm_ihog_kron_ablation as KA
import rpm_ihog_round3 as R3

# ------------------------------------------------------------------------------
# Block-diagonal Hadamard so non-power-of-2 widths (e.g. 1536 = 3 x 512) work.
# Orthonormal, symmetric, self-inverse, exactly like the full Sylvester one.
# ------------------------------------------------------------------------------
_HC = {}


def block_hadamard(n, device, dtype=torch.float32):
    key = (n, str(device), dtype)
    if key not in _HC:
        b = n & (-n)
        H = V1.get_hadamard_matrix(b, device, torch.float32)
        if b != n:
            H = torch.block_diag(*([H] * (n // b)))
        _HC[key] = H.to(dtype)
    return _HC[key]


KA.get_hadamard_matrix = block_hadamard  # ExpertBank.rhat / repack look this up at call time


# ------------------------------------------------------------------------------
# 2-bit + 1-bit stacked residual bank (v8 recipe) on top of the round-3 bank
# ------------------------------------------------------------------------------
class Bank21(R3.Bank):
    def __init__(self, E, d_out, d_in, rank, res="2b1b"):
        super().__init__(E, d_out, d_in, rank, "2bit")          # packed (2-bit codes) + gain
        self.res = "2b1b"                                       # KA._codes falls through to 2-bit
        self.register_buffer("packed1", torch.zeros(E * d_out * d_in // 8, dtype=torch.uint8))
        self.gain1 = nn.Parameter(torch.ones(E, d_out, 1, dtype=torch.float16))
        self._rhat1 = None

    @torch.no_grad()
    def rhat1(self):
        if self._rhat1 is None:
            shape = (self.E, self.d_out, self.d_in)
            q = V1.unpack_1bit_signs_from_uint8(self.packed1, shape, torch.float32)
            Ho, Hi = block_hadamard(self.d_out, self._dev()), block_hadamard(self.d_in, self._dev())
            self._rhat1 = (Ho @ q @ Hi).half()
        return self._rhat1

    def synthesize(self, dtype=torch.float32):
        K = self.kron(dtype)
        W = (K if K is not None else 0.0) + self.gain.to(dtype) * self.rhat().to(dtype) \
            + self.gain1.to(dtype) * self.rhat1().to(dtype)
        return W * torch.exp(self.log_r.to(dtype)) * torch.exp(self.log_c.to(dtype))

    @torch.no_grad()
    def repack(self, W):
        dev = W.device
        scale = torch.exp(self.log_r.float()) * torch.exp(self.log_c.float())
        K = self.kron(torch.float32)
        res = W.float() / scale - (K if K is not None else 0.0)
        Ho, Hi = block_hadamard(self.d_out, dev), block_hadamard(self.d_in, dev)
        x = Ho @ res @ Hi
        sig = x.std(dim=(1, 2), keepdim=True) + 1e-12
        lv = torch.tensor(getattr(KA, "LEVELS_2B", (-1.510, -0.453, 0.453, 1.510)), device=dev)
        idx = torch.argmin(((x / sig).unsqueeze(-1) - lv).abs(), dim=-1)
        rem = x - lv[idx] * sig
        self.packed.copy_(V1.pack_2bit_indices_to_uint8(idx))
        self.packed1.copy_(V1.pack_1bit_signs_to_uint8(torch.where(rem >= 0, 1.0, -1.0)))
        del x, rem, idx
        self._rhat, self._rhat1 = None, None
        R2, R1 = self.rhat().float(), self.rhat1().float()
        # joint least-squares row gains for the two stacked codes (2x2 normal equations per row)
        a, b, c = (R2 * R2).sum(2), (R2 * R1).sum(2), (R1 * R1).sum(2)
        u, v = (res * R2).sum(2), (res * R1).sum(2)
        det = (a * c - b * b).clamp_min(1e-12)
        self.gain.copy_(((c * u - b * v) / det).unsqueeze(-1).to(self.gain.dtype))
        self.gain1.copy_(((a * v - b * u) / det).unsqueeze(-1).to(self.gain1.dtype))


class Block21(R3.Block3):
    def __init__(self, E, d_model, d_ffn, top_k, rank, res, whiten=False, expalloc=False, gamma=0.5):
        super().__init__(E, d_model, d_ffn, top_k, rank, "2bit", whiten, expalloc, gamma)
        self.gate = Bank21(E, d_ffn, d_model, rank)
        self.up = Bank21(E, d_ffn, d_model, rank)
        self.down = Bank21(E, d_model, d_ffn, rank)


CONFIGS = {
    "kron13_1bit": {"tag": "kron13_1bit", "rank": 13, "res": "1bit"},
    "kron13_2bit": {"tag": "kron13_2bit", "rank": 13, "res": "2bit"},
    "kron13_2b1b": {"tag": "kron13_2b1b", "rank": 13, "res": "2b1b", "block_cls": Block21},
}


# ------------------------------------------------------------------------------
# Model plumbing: MoE or dense, any Granite-family layout
# ------------------------------------------------------------------------------
def ffn_attr(layer):
    for a in ("block_sparse_moe", "mlp"):
        if hasattr(layer, a):
            return a
    raise ValueError(f"no FFN on {type(layer)}")


def get_ffn(model, l):
    L = model.model.layers[l]
    return getattr(L, ffn_attr(L))


def set_ffn(model, l, m):
    L = model.model.layers[l]
    setattr(L, ffn_attr(L), m)


def is_dense(m):
    return all(hasattr(m, k) for k in ("gate_proj", "up_proj", "down_proj"))


def extract(m, dev):
    if is_dense(m):
        w_in = torch.cat([m.gate_proj.weight, m.up_proj.weight], 0)[None].float()
        w_out = m.down_proj.weight[None].float()
        rw, tk = torch.zeros(1, w_in.shape[-1]), 1      # one "expert", always selected with prob 1
    else:
        rw, w_in, w_out, tk = V1.extract_granite_moe_weights(m)
    return rw.to(dev), w_in.to(dev), w_out.to(dev), tk


def ffn_params(m):
    if is_dense(m):
        return [m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight]
    if hasattr(m, "input_linear"):
        return [m.input_linear.weight, m.output_linear.weight]
    ex = m.experts
    if hasattr(ex, "gate_up_proj"):
        return [ex.gate_up_proj, ex.down_proj]
    return [p for e in ex for p in (e.gate_proj.weight, e.up_proj.weight, e.down_proj.weight)]


@torch.no_grad()
def rtn_(p, bits, g=64):
    W = p.data
    Wg = W.float().reshape(-1, g)
    qmax = 2 ** (bits - 1) - 1
    s = Wg.abs().amax(1, keepdim=True).clamp_min(1e-8) / qmax
    p.data.copy_((torch.round(Wg / s).clamp(-qmax - 1, qmax) * s).reshape(W.shape).to(W.dtype))


# ------------------------------------------------------------------------------
# Eval with per-sequence NLL -> PPL and bootstrap CI
# ------------------------------------------------------------------------------
@torch.no_grad()
def seq_nll(model, tokens, dev, bs=4):
    model.eval()
    out = []
    for i in range(0, tokens.shape[0], bs):
        b = tokens[i : i + bs].to(dev)
        lg = model(b, use_cache=False).logits[:, :-1].float()
        out.append(F.cross_entropy(lg.transpose(1, 2), b[:, 1:], reduction="none").mean(1))
    return torch.cat(out)


def ppl_stats(nll, ref, n_boot=4000):
    g = torch.Generator(device=nll.device).manual_seed(0)
    n = nll.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g, device=nll.device)
    d = nll[idx].mean(1).exp() - ref[idx].mean(1).exp()
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975], device=d.device)).tolist()
    return {"ppl": round(float(nll.mean().exp()), 3),
            "dppl": round(float(nll.mean().exp() - ref.mean().exp()), 3),
            "ci95": [round(lo, 3), round(hi, 3)]}


# ------------------------------------------------------------------------------
# Storage accounting incl. quantized Kronecker factors
# ------------------------------------------------------------------------------
def bpw(blk, factor_bits=16):
    n = blk.E * 3 * blk.gate.d_out * blk.gate.d_in
    bits = 0
    for k, v in blk.state_dict().items():
        name = k.split(".")[-1]
        if k == "router_w":
            continue
        if name in ("A", "B"):
            bank = getattr(blk, k.split(".")[0])
            active = bank.rank_mask.sum().item()
            per = v[0, 0].numel()
            bits += active * (factor_bits * per + (16 if factor_bits < 16 else 0))
        elif name == "rank_mask":
            bits += 8 * v.shape[0]
        else:
            bits += 8 * v.numel() * v.element_size()
    return bits / n


@torch.no_grad()
def quantize_factors_(blocks, bits):
    saved = []
    qmax = 2 ** (bits - 1) - 1
    for blk in blocks:
        for b in blk.banks():
            if b.rank == 0:
                continue
            for P in (b.A, b.B):
                saved.append((P, P.data.clone()))
                X = P.data.float()
                s = X.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1e-12) / qmax
                P.data.copy_((torch.round(X / s).clamp(-qmax - 1, qmax) * s).to(P.dtype))
    return saved


def restore_(saved):
    with torch.no_grad():
        for P, d in saved:
            P.data.copy_(d)


# ------------------------------------------------------------------------------
# Closed-loop chain over any set of layers; teacher activations captured lazily
# (all-original model) so nothing is cached for layers that aren't being worked on.
# ------------------------------------------------------------------------------
def teacher_for(model, orig, replaced, l, calib, evalt, dev):
    cur = {i: get_ffn(model, i) for i in replaced}
    for i in replaced:
        set_ffn(model, i, orig[i])
    try:
        Xc, Yc, rt = R3.capture_real_moe_io(model, orig[l], calib, dev)
        Xe, Ye, _ = R3.capture_real_moe_io(model, orig[l], evalt, dev)
    finally:
        for i in replaced:
            set_ffn(model, i, cur[i])
    rw, wi, wo, tk = extract(orig[l], dev)
    return dict(X_calib_clean=Xc, Y_calib_teacher=Yc, X_eval_clean=Xe, Y_eval_teacher=Ye,
                returns_tuple=rt, router_w=rw, w_in=wi, w_out=wo, top_k=tk)


def run_chain(name, specs, model, orig, layers, calib, evalt, dev, args):
    """Compress `layers` in order (closed loop). Leaves them compressed in the model."""
    replaced, per = [], {}
    for l in layers:
        t = teacher_for(model, orig, replaced, l, calib, evalt, dev)
        if replaced:
            Xci, _, _ = R3.capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
            Xei, _, _ = R3.capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
        else:
            Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
        blk, m = R3.train_layer(name, specs[l], t, Xci, Xei, dev, args,
                                os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"), args.steps)
        set_ffn(model, l, blk)
        replaced.append(l)
        m["bpw"] = round(bpw(blk), 4)
        m["tag"] = specs[l]["tag"]
        per[l] = m
        print(f"  L{l:2d} [{specs[l]['tag']:<11}] bpw={m['bpw']:.4f} oneshot={m['oneshot_r2']:6.2f}%  "
              f"closed={m['closed_r2']:6.2f}%  open={m['open_r2']:6.2f}%  ({m['train_sec']}s)", flush=True)
        del t, Xci, Xei
        torch.cuda.empty_cache()
    return per


def restore_model(model, orig, layers):
    for l in layers:
        set_ffn(model, l, orig[l])


def mean_bpw(model, layers, fb=16):
    return round(sum(bpw(get_ffn(model, l), fb) for l in layers) / len(layers), 4)


def avg(per, k):
    return round(sum(m[k] for m in per.values()) / len(per), 2)


# ------------------------------------------------------------------------------
# Allocation (full depth): Lagrangian sweep + greedy top-up. Scales to any depth.
# ------------------------------------------------------------------------------
def lagrange_alloc(sens, layers, ladder, budget):
    n, best = len(layers), None
    for lam in [0.0] + [10 ** (k / 40) for k in range(-320, 81)]:
        pick = [min(range(len(ladder)), key=lambda c, l=l: sens[l][c]["dnll"] + lam * sens[l][c]["bpw"])
                for l in layers]
        b = sum(sens[l][c]["bpw"] for l, c in zip(layers, pick)) / n
        if b <= budget + 1e-6:
            cost = sum(sens[l][c]["dnll"] for l, c in zip(layers, pick))
            if best is None or cost < best[0]:
                best = (cost, b, pick)
    if best is None:
        raise RuntimeError(f"no allocation fits budget {budget}")
    cost, b, pick = best
    while True:  # spend leftover budget on the single best-improving upgrade, repeatedly
        cand = None
        for i, l in enumerate(layers):
            for c2 in range(len(ladder)):
                db = (sens[l][c2]["bpw"] - sens[l][pick[i]]["bpw"]) / n
                dd = sens[l][c2]["dnll"] - sens[l][pick[i]]["dnll"]
                if dd < 0 and b + db <= budget + 1e-6 and (cand is None or dd < cand[0]):
                    cand = (dd, i, c2, db)
        if cand is None:
            break
        dd, i, c2, db = cand
        pick[i], b, cost = c2, b + db, cost + dd
    return cost, b, pick


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--exps", nargs="+", default=["rtn", "noise", "frontier"],
                    choices=["rtn", "noise", "frontier", "fullalloc"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--layers", type=int, nargs="+", default=None, help="default: ALL layers")
    ap.add_argument("--noise_layers", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rtn_bits", type=int, nargs="+", default=[8, 4, 3, 2])
    ap.add_argument("--factor_bits", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--teacher_quant", type=int, default=0, help="8 or 4: compress an RTN-quantized model")
    ap.add_argument("--budget", type=float, default=None, help="fullalloc mean bpw (default: kron13_2bit's)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sens_steps", type=int, default=150)
    ap.add_argument("--sens_seqs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--batch_tokens", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    args.out_dir = args.out_dir or f"./rpm_ihog_round4_{args.model_id.split('/')[-1]}"
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, path = V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    with open(path, encoding="utf-8") as f:
        print(f"[*] Corpus: {'TinyShakespeare' if 'First Citizen' in f.read(4000) else 'WikiText-2'} ({path})")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nL = len(model.model.layers)
    layers = args.layers or list(range(nL))
    orig = {l: get_ffn(model, l) for l in range(nL)}
    kind = "dense" if is_dense(orig[0]) else "MoE"
    n_ffn = sum(p.numel() for l in range(nL) for p in ffn_params(orig[l]))
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[*] {args.model_id}: {kind}, {nL} layers, FFN params {n_ffn/1e9:.2f}B of {n_all/1e9:.2f}B "
          f"({100*n_ffn/n_all:.0f}%), compressing {len(layers)} layers")

    nll_fp16 = seq_nll(model, evalt, dev)
    print(f"[*] FP16 PPL {float(nll_fp16.mean().exp()):.3f}")
    results = {"model": args.model_id, "kind": kind, "fp16_ppl": round(float(nll_fp16.mean().exp()), 3),
               "args": vars(args), "rows": []}

    def save():
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    def row(name, bpw_, nll, ref_teacher=None, **extra):
        r = {"name": name, "bpw": bpw_, **ppl_stats(nll, nll_fp16), **extra}
        if ref_teacher is not None:
            r["dppl_vs_teacher"] = ppl_stats(nll, ref_teacher)["dppl"]
        results["rows"].append(r)
        print(f"  -> {name:<26} bpw={bpw_:<7} PPL {r['ppl']} (Δ {r['dppl']:+} CI {r['ci95']})", flush=True)
        save()
        return r

    if "rtn" in args.exps or args.teacher_quant:
        backup = [p.data.cpu().clone() for l in range(nL) for p in ffn_params(orig[l])]

        def put_back():
            with torch.no_grad():
                for p, d in zip((p for l in range(nL) for p in ffn_params(orig[l])), backup):
                    p.data.copy_(d.to(p.device))

    if "rtn" in args.exps:
        print("\n=== rtn (all FFN layers, group 64) ===")
        for bits in args.rtn_bits:
            for l in range(nL):
                for p in ffn_params(orig[l]):
                    rtn_(p, bits)
            row(f"rtn_int{bits}_g64", bits + 0.25, seq_nll(model, evalt, dev))
            put_back()

    ref_teacher = None
    if args.teacher_quant:
        for l in range(nL):
            for p in ffn_params(orig[l]):
                rtn_(p, args.teacher_quant)
        ref_teacher = seq_nll(model, evalt, dev)
        row(f"teacher_rtn_int{args.teacher_quant}", args.teacher_quant + 0.25, ref_teacher)
        print(f"[*] Compressing the INT{args.teacher_quant} model from here on.")

    if "noise" in args.exps:
        nl = [l for l in args.noise_layers if l < nL]
        print(f"\n=== noise: kron13_1bit on layers {nl}, seeds {args.seeds} ===")
        d = []
        for s in args.seeds:
            args.seed = s
            run_chain(f"noise_s{s}", {l: CONFIGS["kron13_1bit"] for l in nl}, model, orig, nl, calib, evalt, dev, args)
            r = row(f"noise_seed{s}", mean_bpw(model, nl), seq_nll(model, evalt, dev), ref_teacher)
            d.append(r["dppl"])
            restore_model(model, orig, nl)
        m = sum(d) / len(d)
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1))
        results["noise_seed_std_dppl"] = round(sd, 4)
        print(f"  seed-to-seed dPPL: mean {m:+.3f}, std {sd:.3f}")
        args.seed = 42
        save()

    if "frontier" in args.exps:
        for c in args.configs:
            print(f"\n=== frontier: {c} on {len(layers)} layers ===")
            per = run_chain(c, {l: CONFIGS[c] for l in layers}, model, orig, layers, calib, evalt, dev, args)
            blocks = [get_ffn(model, l) for l in layers]
            for fb in args.factor_bits:
                saved = quantize_factors_(blocks, fb) if fb < 16 else []
                row(f"{c}_f{fb}", mean_bpw(model, layers, fb), seq_nll(model, evalt, dev), ref_teacher,
                    avg_open_r2=avg(per, "open_r2"), avg_closed_r2=avg(per, "closed_r2"))
                restore_(saved)
            restore_model(model, orig, layers)
            torch.cuda.empty_cache()

    if "fullalloc" in args.exps:
        ladder = [CONFIGS[k] for k in ("kron13_1bit", "kron13_2bit", "kron13_2b1b")]
        print(f"\n=== fullalloc phase 1: sensitivity of {len(layers)} layers x {len(ladder)} configs ===")
        sens_tok = calib[: args.sens_seqs]
        ref = seq_nll(model, sens_tok, dev).mean().item()
        sens = {}
        for l in layers:
            t = teacher_for(model, orig, [], l, calib, evalt, dev)
            sens[l] = []
            for spec in ladder:
                blk, m = R3.train_layer("sens", spec, t, t["X_calib_clean"], t["X_eval_clean"], dev, args,
                                        os.path.join(args.out_dir, "ckpt_sens.pt"), args.sens_steps)
                set_ffn(model, l, blk)
                dn = seq_nll(model, sens_tok, dev).mean().item() - ref
                set_ffn(model, l, orig[l])
                sens[l].append({"bpw": bpw(blk), "dnll": dn})
                print(f"  L{l:2d} {spec['tag']:<11} bpw={sens[l][-1]['bpw']:.4f} dNLL={dn:+.5f}", flush=True)
            del t
        budget = args.budget or sum(s[1]["bpw"] for s in sens.values()) / len(layers)
        cost, b, pick = lagrange_alloc(sens, layers, ladder, budget)
        specs = {l: ladder[c] for l, c in zip(layers, pick)}
        print(f"=== fullalloc phase 2: budget {budget:.4f} -> mean bpw {b:.4f}; "
              + " ".join(f"L{l}:{specs[l]['tag'].split('_')[1]}" for l in layers))
        per = run_chain("fullalloc", specs, model, orig, layers, calib, evalt, dev, args)
        blocks = [get_ffn(model, l) for l in layers]
        for fb in args.factor_bits:
            saved = quantize_factors_(blocks, fb) if fb < 16 else []
            row(f"fullalloc_f{fb}", mean_bpw(model, layers, fb), seq_nll(model, evalt, dev), ref_teacher,
                avg_open_r2=avg(per, "open_r2"))
            restore_(saved)
        results["fullalloc_sens"] = {str(l): v for l, v in sens.items()}
        restore_model(model, orig, layers)
        save()

    print(f"\n{'name':<26}{'bpw':>8}{'PPL':>9}{'ΔPPL':>9}   {'95% CI':<18}{'openR2':>8}")
    for r in sorted(results["rows"], key=lambda r: r["bpw"]):
        print(f"{r['name']:<26}{r['bpw']:>8.3f}{r['ppl']:>9.3f}{r['dppl']:>+9.3f}   "
              f"{str(r['ci95']):<18}{str(r.get('avg_open_r2', '')):>8}")
    print(f"[+] {os.path.join(args.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
