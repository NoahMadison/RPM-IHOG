#!/usr/bin/env python3
r"""
rpm_ihog_round3.py
==================
Three separate experiments, each compared against the round-2 winner `kron13_1bit`
(rank-13 per-expert Kronecker + 1-bit Hadamard residual, ~1.68 bpw), same closed-loop
protocol, same WikiText corpus as v1.

  base       : kron13_1bit, plain Kronecker SVD init (reference)
  wsvd       : kron13_1bit, ACTIVATION-WHITENED Kronecker SVD init.
               Input covariance per expert is approximated as Ca (x) Cb (nearest Kronecker
               product), so whitening stays Kronecker-structured:
                   (A (x) B)(Sa (x) Sb) = (A Sa) (x) (B Sb)
               SVD is done in whitened space, then un-whitened back into A, B.
               Same storage as base; only the starting point changes.
  expalloc   : kron13_1bit with Kronecker rank REDISTRIBUTED across experts by routed
               token mass (rank_e ~ usage_e^gamma), total rank held at 13 * E.
               Same total bpw as base. The expert-level "redistribute the budget to where
               the activity is" experiment.
  layeralloc : per-layer bit allocation. Ladder {kron6_1bit, kron13_1bit, kron9_tern}.
               Phase 1: compress each layer ALONE (open loop, fewer steps), measure dNLL on
                        CALIBRATION tokens (not eval -> no selection leakage).
               Phase 2: exhaustive search over ladder^layers (3^5 = 243 combos, instant),
                        min sum dNLL s.t. mean bpw <= budget (default = base bpw).
               Phase 3: closed-loop run with the chosen per-layer configs.

Everything (activations, teacher weights, batch sampling) stays on the GPU.
Requires rpm_ihog_kron_ablation.py and rpm_ihog_15bpw_hadamard_distill.py in the same dir.
"""

import argparse
import itertools
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rpm_ihog_15bpw_hadamard_distill import (
    evaluate_full_model_heldout,
    extract_granite_moe_weights,
    load_disjoint_calib_and_eval_tokens,
)
from rpm_ihog_kron_ablation import Block, ExpertBank  # model classes only; identical in every version


# ==============================================================================
# Self-contained GPU-resident capture + eval (don't depend on whichever ablation copy is on disk)
# ==============================================================================
@torch.no_grad()
def capture_real_moe_io(model, moe, tokens, dev, batch_size=4, capture_outputs=True):
    xs, ys, rt = [], [], [False]

    def hook(m, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        rt[0] = rt[0] or isinstance(out, tuple)
        xs.append(inp[0].detach().reshape(-1, inp[0].shape[-1]).half())
        if capture_outputs:
            ys.append(o.detach().reshape(-1, o.shape[-1]).half())

    h = moe.register_forward_hook(hook)
    model.eval()
    try:
        for i in range(0, tokens.shape[0], batch_size):
            model(tokens[i : i + batch_size].to(dev), use_cache=False)
    finally:
        h.remove()
    return torch.cat(xs).to(dev), (torch.cat(ys).to(dev) if capture_outputs else None), rt[0]


@torch.no_grad()
def eval_block(blk, X, Y, dev, chunk=2048):
    blk.eval()
    X, T = X.to(dev), Y.to(dev).float()
    P = torch.cat([blk.forward_moe(X[i : i + chunk].float()) for i in range(0, X.shape[0], chunk)]).float()
    mse = (P - T).pow(2).mean().item()
    return {
        "r2_pct": round(100.0 * (1.0 - mse / (T.var().item() + 1e-12)), 2),
        "cos": round(F.cosine_similarity(P.flatten(), T.flatten(), dim=0).item(), 5),
    }


# ==============================================================================
# Bank with (a) per-expert rank mask, (b) Kronecker-factored whitened init
# ==============================================================================
def _psd_sqrt(C, damp):
    C = 0.5 * (C + C.transpose(-1, -2))
    lam, V = torch.linalg.eigh(C)
    floor = damp * lam.mean(-1, keepdim=True).clamp_min(1e-12)
    lam = torch.maximum(lam, floor)
    s = lam.sqrt()
    S = (V * s.unsqueeze(-2)) @ V.transpose(-1, -2)
    Si = (V / s.unsqueeze(-2)) @ V.transpose(-1, -2)
    return S, Si


class Bank(ExpertBank):
    def __init__(self, E, d_out, d_in, rank, res):
        super().__init__(E, d_out, d_in, rank, res)
        if rank > 0:
            # 1 = active Kronecker term. Serialized so a reload reproduces the allocation.
            self.register_buffer("rank_mask", torch.ones(E, rank, dtype=torch.float16))

    def kron(self, dtype):
        if self.rank == 0:
            return None
        A = self.A.to(dtype) * self.rank_mask.to(dtype)[:, :, None, None]
        return torch.einsum("erij,erkl->eikjl", A, self.B.to(dtype)).reshape(self.E, self.d_out, self.d_in)

    @torch.no_grad()
    def init_from_teacher(self, W, act_rms, cov=None, damp=0.01):
        if cov is None or self.rank == 0:
            return super().init_from_teacher(W, act_rms)

        E, p1, p2, q1, q2, r = self.E, self.p1, self.p2, self.q1, self.q2, self.rank
        a = act_rms.reshape(-1, 1, self.d_in).float() + 1e-6
        W_act = W.float() * a
        rs = W_act.pow(2).mean(2, keepdim=True).sqrt() + 1e-8
        cs = (W_act / rs).pow(2).mean(1, keepdim=True).sqrt() + 1e-8
        W_bal = W_act / (rs * cs)
        c_s = cs / a                                   # W = K * rs * c_s  ->  K sees z = c_s * x
        self.log_r.copy_(torch.log(rs).half())
        self.log_c.copy_(torch.log(c_s).half())

        # Covariance of z, then nearest Kronecker product Cz ~ Ca (x) Cb (in index = j*q2 + l)
        Cz = cov.float() * c_s.transpose(1, 2) * c_s           # (E, d_in, d_in)
        Rm = Cz.reshape(E, q1, q2, q1, q2).permute(0, 1, 3, 2, 4).reshape(E, q1 * q1, q2 * q2)
        U, S, Vh = torch.linalg.svd(Rm, full_matrices=False)
        Ca = (U[:, :, 0] * S[:, :1].sqrt()).reshape(E, q1, q1)
        Cb = (Vh[:, 0, :] * S[:, :1].sqrt()).reshape(E, q2, q2)
        sgn = torch.sign(Ca.diagonal(dim1=1, dim2=2).sum(-1)).reshape(E, 1, 1)
        Ca, Cb = Ca * sgn, Cb * sgn
        Sa, Sa_i = _psd_sqrt(Ca, damp)
        Sb, Sb_i = _psd_sqrt(Cb, damp)

        # Whitened target, Kronecker SVD, un-whiten into A and B
        W5 = W_bal.reshape(E, p1, p2, q1, q2)                  # [e, i, k, j, l]
        M5 = torch.einsum("eikjl,ejJ,elL->eikJL", W5, Sa, Sb)
        Km = M5.permute(0, 1, 3, 2, 4).reshape(E, p1 * q1, p2 * q2)
        U, S, Vh = torch.linalg.svd(Km, full_matrices=False)
        s = S[:, :r].sqrt()
        A_w = (U[:, :, :r] * s[:, None, :]).permute(0, 2, 1).reshape(E, r, p1, q1)
        B_w = (Vh[:, :r, :] * s[:, :, None]).reshape(E, r, p2, q2)
        self.A.copy_(torch.einsum("eriJ,eJj->erij", A_w, Sa_i).half())
        self.B.copy_(torch.einsum("erkL,eLl->erkl", B_w, Sb_i).half())

        self.repack(W)


class Block3(Block):
    def __init__(self, E, d_model, d_ffn, top_k, rank, res, whiten=False, expalloc=False, gamma=0.5):
        self.base_rank = rank
        kron_cap = (d_ffn // 32) * (d_model // 32)          # max Kronecker rank the SVD can supply
        rmax = min(2 * rank, kron_cap) if expalloc else rank
        super().__init__(E, d_model, d_ffn, top_k, rmax, res)
        self.gate = Bank(E, d_ffn, d_model, rmax, res)
        self.up = Bank(E, d_ffn, d_model, rmax, res)
        self.down = Bank(E, d_model, d_ffn, rmax, res)
        self.whiten, self.expalloc, self.gamma = whiten, expalloc, gamma
        self.info = {}

    @torch.no_grad()
    def init_from_teacher(self, router_w, w_in, w_out, X, Y, dev):
        self.router_w.copy_(router_w.half().to(dev))
        wg, wu = w_in.to(dev).float().chunk(2, dim=1)
        wd = w_out.to(dev).float()
        x = X.to(dev).float()
        N, E = x.shape[0], self.E
        rms_in = x.pow(2).mean(0).sqrt()

        probs = F.softmax(F.linear(x, self.router_w.float()), dim=-1)
        tp, ti = torch.topk(probs, self.top_k, dim=-1)
        tp = tp / tp.sum(-1, keepdim=True)
        mass = torch.zeros(E, device=dev).index_add_(0, ti.flatten(), tp.flatten())
        usage = mass / mass.sum()
        self.info["usage_entropy_norm"] = round(float(-(usage * usage.clamp_min(1e-12).log()).sum() / math.log(E)), 4)
        self.info["usage_max_over_min"] = round(float(usage.max() / usage.min().clamp_min(1e-12)), 2)

        mids, cin, cmid = [], [], []
        for e in range(E):
            tok, kp = (ti == e).nonzero(as_tuple=True)
            w = tp[tok, kp]
            if tok.numel() < 64:
                tok = torch.arange(min(N, 1024), device=dev)
                w = torch.ones(tok.numel(), device=dev)
            xe = x[tok]
            h = F.silu(xe @ wg[e].T) * (xe @ wu[e].T)
            mids.append(h.pow(2).mean(0).sqrt())
            if self.whiten:
                w2 = (w * w)[:, None]                  # expert output is scaled by its gate prob
                cin.append((xe * w2).T @ xe / w2.sum())
                cmid.append((h * w2).T @ h / w2.sum())
        rms_mid = torch.stack(mids)

        if self.expalloc and self.gate.rank > 0:
            rmax, target = self.gate.rank, self.base_rank * E
            raw = usage.pow(self.gamma)
            raw = raw / raw.sum() * target
            alloc = raw.floor().clamp(2, rmax)
            while alloc.sum() < target:
                gap = torch.where(alloc < rmax, raw - alloc, torch.full_like(raw, -1e9))
                alloc[gap.argmax()] += 1
            while alloc.sum() > target:
                gap = torch.where(alloc > 2, raw - alloc, torch.full_like(raw, 1e9))
                alloc[gap.argmin()] -= 1
            mask = (torch.arange(rmax, device=dev)[None, :] < alloc[:, None]).half()
            for b in self.banks():
                b.rank_mask.copy_(mask)
            self.info["rank_min_max"] = [int(alloc.min()), int(alloc.max())]

        cov_in = torch.stack(cin) if self.whiten else None
        cov_mid = torch.stack(cmid) if self.whiten else None
        self.gate.init_from_teacher(wg, rms_in, cov_in)
        self.up.init_from_teacher(wu, rms_in, cov_in)
        self.down.init_from_teacher(wd, rms_mid, cov_mid)

        xs, ys = x[:4096], Y[:4096].to(dev).float()
        self.out_gain.fill_(1.0)
        yp = self.forward_moe(xs)
        alpha = float((yp * ys).sum() / (yp.pow(2).sum() + 1e-12))
        self.out_gain.fill_(min(max(alpha, 0.05), 20.0))


def eff_bpw(blk):
    """Honest storage: only active Kronecker terms count (+1 byte/expert for the rank count)."""
    n = blk.E * 3 * blk.gate.d_out * blk.gate.d_in
    bits = 16  # out_gain
    for b in blk.banks():
        bits += 16 * (b.log_r.numel() + b.log_c.numel())
        if b.rank > 0:
            bits += 16 * b.rank_mask.sum().item() * (b.p1 * b.q1 + b.p2 * b.q2) + 8 * b.E
        if b.res != "none":
            bits += 8 * b.packed.numel() + 16 * b.gain.numel()
    return round(bits / n, 4)


# ==============================================================================
# Training (same recipe as the ablation; block construction is pluggable)
# ==============================================================================
def train_layer(name, spec, t, X_cal_in, X_ev_in, dev, args, ckpt, steps):
    E, two_f, d = t["w_in"].shape

    def mk():
        b = Block3(E, d, two_f // 2, t["top_k"], spec["rank"], spec["res"],
                   whiten=spec.get("whiten", False), expalloc=spec.get("expalloc", False),
                   gamma=args.gamma).to(dev)
        b._return_tuple = t["returns_tuple"]
        return b

    blk = mk()
    blk.init_from_teacher(t["router_w"], t["w_in"], t["w_out"], t["X_calib_clean"], t["Y_calib_teacher"], dev)
    oneshot = eval_block(blk, X_ev_in, t["Y_eval_teacher"], dev)

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
        betas=(0.9, 0.95), weight_decay=0.0,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    X_clean, Y = t["X_calib_clean"], t["Y_calib_teacher"]
    N = X_cal_in.shape[0]
    var = Y.float().var().item() + 1e-12
    gen = torch.Generator(device=dev).manual_seed(42)
    repack_at = {int(0.30 * steps), int(0.60 * steps)}
    t0, act_val = time.perf_counter(), float("nan")

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (args.batch_tokens,), generator=gen, device=dev)
        x, xc, y = X_cal_in[idx].float(), X_clean[idx].float(), Y[idx].float()
        Ws = blk.synth_all(torch.float32)
        yp = blk.forward_moe(x, Ws, router_x=xc if step < int(0.65 * steps) else None)
        act = F.mse_loss(yp, y) / var
        wl = sum(((W - T).pow(2).mean() / P) for W, T, P in zip(Ws, teach, teach_pow))
        loss = act + args.anchor * wl
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{name}] non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        act_val = act.item()
        if spec["res"] != "none" and step in repack_at:
            for b, T in zip(blk.banks(), teach):
                b.repack(T)
                opt.state.pop(b.gain, None)

    with torch.no_grad():
        for p in params:
            p.data = p.data.half()
    dt = time.perf_counter() - t0

    torch.save(blk.state_dict(), ckpt)
    re = mk()
    re.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True), strict=True)
    re.eval()
    closed = eval_block(re, X_ev_in, t["Y_eval_teacher"], dev)
    opened = eval_block(re, t["X_eval_clean"], t["Y_eval_teacher"], dev)
    return re, {
        "bpw": eff_bpw(re), "oneshot_r2": oneshot["r2_pct"],
        "closed_r2": closed["r2_pct"], "open_r2": opened["r2_pct"],
        "final_calib_nmse": round(act_val, 5), "train_sec": round(dt, 1), **blk.info,
    }


# ==============================================================================
# Runners
# ==============================================================================
def run_closed_loop(name, specs, model, orig, teacher, calib, evalt, base_top1, base_ppl, dev, args):
    """specs: {layer: spec}. Layers compressed in order; each sees the drifted inputs."""
    print(f"\n=== {name} ===")
    for l in orig:
        model.model.layers[l].block_sparse_moe = orig[l]
    per = {}
    for i, l in enumerate(args.layers):
        t = teacher[l]
        if i == 0:
            Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
        else:
            Xci, _, _ = capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
            Xei, _, _ = capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
            Xci, Xei = Xci.to(dev), Xei.to(dev)
        blk, m = train_layer(name, specs[l], t, Xci, Xei, dev, args,
                             os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"), args.steps)
        model.model.layers[l].block_sparse_moe = blk
        m["spec"] = specs[l]
        per[l] = m
        extra = f" ranks={m['rank_min_max']}" if "rank_min_max" in m else ""
        print(f"  L{l:2d} [{specs[l]['tag']:<11}] bpw={m['bpw']:.4f} oneshot={m['oneshot_r2']:6.2f}%  "
              f"closed={m['closed_r2']:6.2f}%  open={m['open_r2']:6.2f}%{extra}")
    fm, _ = evaluate_full_model_heldout(model, evalt, dev, baseline_top1=base_top1)
    for l in orig:
        model.model.layers[l].block_sparse_moe = orig[l]
    avg = lambda k: round(sum(m[k] for m in per.values()) / len(per), 2)
    res = {
        "bpw": round(sum(m["bpw"] for m in per.values()) / len(per), 4),
        "avg_oneshot_r2": avg("oneshot_r2"), "avg_closed_r2": avg("closed_r2"), "avg_open_r2": avg("open_r2"),
        "ppl": fm["heldout_ppl"], "ppl_delta": round(fm["heldout_ppl"] - base_ppl, 3),
        "top1": fm["heldout_top1_agreement_pct"], "per_layer": per,
    }
    print(f"  -> mean bpw {res['bpw']} | PPL {res['ppl']} (Δ {res['ppl_delta']:+}) | top1 {res['top1']}%")
    return res


LADDER = [
    {"tag": "kron6_1bit", "rank": 6, "res": "1bit"},
    {"tag": "kron13_1bit", "rank": 13, "res": "1bit"},
    {"tag": "kron9_tern", "rank": 9, "res": "tern"},
]


def choose_allocation(sens, layers, budget):
    """Exhaustive search: minimize sum dNLL subject to mean bpw <= budget. Assumes additivity."""
    best = None
    for combo in itertools.product(range(len(LADDER)), repeat=len(layers)):
        bpw = sum(sens[l][c]["bpw"] for l, c in zip(layers, combo)) / len(layers)
        if bpw > budget + 1e-6:
            continue
        cost = sum(sens[l][c]["dnll"] for l, c in zip(layers, combo))
        if best is None or cost < best[0]:
            best = (cost, bpw, combo)
    if best is None:
        raise RuntimeError(f"no allocation fits budget {budget}")
    return best


def run_layeralloc(model, orig, teacher, calib, evalt, base_top1, base_ppl, budget, dev, args):
    print("\n=== layeralloc: phase 1 (per-layer sensitivity, open loop, calib tokens) ===")
    sens_tokens = calib[: args.sens_seqs]
    for l in orig:
        model.model.layers[l].block_sparse_moe = orig[l]
    ref, _ = evaluate_full_model_heldout(model, sens_tokens, dev)
    sens = {}
    for l in args.layers:
        t, sens[l] = teacher[l], []
        for c, spec in enumerate(LADDER):
            blk, m = train_layer(f"sens_{spec['tag']}", spec, t, t["X_calib_clean"], t["X_eval_clean"], dev, args,
                                 os.path.join(args.out_dir, f"ckpt_sens_L{l}_{spec['tag']}.pt"), args.sens_steps)
            model.model.layers[l].block_sparse_moe = blk
            ev, _ = evaluate_full_model_heldout(model, sens_tokens, dev)
            model.model.layers[l].block_sparse_moe = orig[l]
            d = ev["heldout_nll"] - ref["heldout_nll"]
            sens[l].append({"bpw": m["bpw"], "dnll": d, "open_r2": m["open_r2"]})
            print(f"  L{l:2d} {spec['tag']:<11} bpw={m['bpw']:.4f} open={m['open_r2']:6.2f}%  dNLL={d:+.5f}")

    cost, bpw, combo = choose_allocation(sens, args.layers, budget)
    chosen = {l: LADDER[c] for l, c in zip(args.layers, combo)}
    print(f"\n=== layeralloc: phase 2 -> budget {budget:.4f}, chosen mean bpw {bpw:.4f}, "
          f"predicted sum dNLL {cost:+.5f}")
    for l in args.layers:
        print(f"  L{l:2d} -> {chosen[l]['tag']}")

    res = run_closed_loop("layeralloc", chosen, model, orig, teacher, calib, evalt, base_top1, base_ppl, dev, args)
    res["sensitivity"] = {str(l): v for l, v in sens.items()}
    res["predicted_sum_dnll"] = cost
    return res


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--exps", nargs="+", default=["base", "wsvd", "expalloc", "layeralloc"],
                    choices=["base", "wsvd", "expalloc", "layeralloc"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sens_steps", type=int, default=150, help="steps per sensitivity probe (ranking only)")
    ap.add_argument("--sens_seqs", type=int, default=16, help="calib sequences used to score sensitivity")
    ap.add_argument("--budget", type=float, default=None, help="mean bpw for layeralloc (default: base bpw)")
    ap.add_argument("--gamma", type=float, default=0.5, help="expalloc: rank_e ~ usage_e^gamma")
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=0.05)
    ap.add_argument("--batch_tokens", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", default="./rpm_ihog_round3")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, path = load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir
    )
    with open(path, encoding="utf-8") as f:
        print(f"[*] Corpus: {'TinyShakespeare' if 'First Citizen' in f.read(4000) else 'WikiText-2'} ({path})")

    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base, base_top1 = evaluate_full_model_heldout(model, evalt, dev)
    print(f"[*] FP16 baseline PPL {base['heldout_ppl']}")

    orig = {l: model.model.layers[l].block_sparse_moe for l in args.layers}
    teacher = {}
    for l in args.layers:
        Xc, Yc, rt = capture_real_moe_io(model, orig[l], calib, dev)
        Xe, Ye, _ = capture_real_moe_io(model, orig[l], evalt, dev)
        Xc, Yc, Xe, Ye = Xc.to(dev), Yc.to(dev), Xe.to(dev), Ye.to(dev)  # no-op if already on GPU
        rw, wi, wo, tk = extract_granite_moe_weights(orig[l])
        teacher[l] = dict(X_calib_clean=Xc, Y_calib_teacher=Yc, X_eval_clean=Xe, Y_eval_teacher=Ye,
                          returns_tuple=rt, router_w=rw.to(dev), w_in=wi.to(dev), w_out=wo.to(dev), top_k=tk)

    ref_spec = {"tag": "kron13_1bit", "rank": 13, "res": "1bit"}
    all_specs = {
        "base": {l: ref_spec for l in args.layers},
        "wsvd": {l: {**ref_spec, "tag": "wsvd", "whiten": True} for l in args.layers},
        "expalloc": {l: {**ref_spec, "tag": "expalloc", "expalloc": True} for l in args.layers},
    }
    results = {"baseline_ppl": base["heldout_ppl"], "args": vars(args), "exps": {}}
    save = lambda: json.dump(results, open(os.path.join(args.out_dir, "results.json"), "w"), indent=2, default=str)

    for name in args.exps:
        if name == "layeralloc":
            budget = args.budget
            if budget is None:
                budget = results["exps"]["base"]["bpw"] if "base" in results["exps"] else 1.6771
            r = run_layeralloc(model, orig, teacher, calib, evalt, base_top1, base["heldout_ppl"], budget, dev, args)
        else:
            r = run_closed_loop(name, all_specs[name], model, orig, teacher, calib, evalt,
                                base_top1, base["heldout_ppl"], dev, args)
        results["exps"][name] = r
        save()

    ref = results["exps"].get("base")
    print(f"\n{'exp':<11}{'bpw':>8}{'oneshotR2':>11}{'closedR2':>10}{'openR2':>9}{'PPL':>9}{'ΔPPL':>9}{'top1':>8}"
          f"{'ΔPPL vs base':>14}")
    for name, r in results["exps"].items():
        vs = f"{r['ppl'] - ref['ppl']:+.3f}" if ref else "-"
        print(f"{name:<11}{r['bpw']:>8.4f}{r['avg_oneshot_r2']:>11.2f}{r['avg_closed_r2']:>10.2f}"
              f"{r['avg_open_r2']:>9.2f}{r['ppl']:>9.3f}{r['ppl_delta']:>+9.3f}{r['top1']:>8.2f}{vs:>14}")


if __name__ == "__main__":
    main()
