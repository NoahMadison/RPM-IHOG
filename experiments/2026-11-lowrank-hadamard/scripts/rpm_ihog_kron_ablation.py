#!/usr/bin/env python3
r"""
rpm_ihog_kron_ablation.py
=========================
Kronecker core vs. no Kronecker core, at matched bpw, on Granite MoE.

Matched pairs (expected exact bpw):
  kron13_1bit  ~1.68   vs   nokron_tern  ~1.67
  kron9_tern   ~2.09   vs   nokron_2bit  ~2.07
References:
  kron13_none  ~0.66   (pure Kronecker)
  nokron_1bit  ~1.07   (what the 1-bit residual does alone)

Fixes vs v2:
  - Scales are log-parameterized (log_r, log_c): always positive, Adam steps are relative.
    Repack divides by exp(log_r)*exp(log_c) -> can never hit the clamp/sign-flip blowup.
  - Scales get their own (lower) LR group.
  - Global output gain is a separate scalar (out_gain), not folded into down.row_s,
    so repacks don't silently re-absorb it.
  - Residual gain is fit by least squares in the domain where it is APPLIED
    (per unrotated row, against H q H). v1/v2 fit it per rotated row and applied it per
    unrotated row.
  - Ternary/2-bit thresholds use a per-expert sigma so encode and decode agree.
  - Adam state for residual gains is reset after each repack.
  - Hard NaN guard.
  - Reports BOTH closed-loop R^2 (drifted inputs) and open-loop R^2 (clean inputs) per layer,
    so cascade error and per-layer difficulty can be separated.

Reuses data/model utilities from rpm_ihog_15bpw_hadamard_distill.py (v1) in the same dir.
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

from rpm_ihog_15bpw_hadamard_distill import (
    capture_real_moe_io,
    evaluate_full_model_heldout,
    extract_granite_moe_weights,
    get_hadamard_matrix,
    load_disjoint_calib_and_eval_tokens,
    pack_1bit_signs_to_uint8,
    pack_2bit_indices_to_uint8,
    pack_ternary_indices_to_uint8,
    unpack_1bit_signs_from_uint8,
    unpack_2bit_indices_from_uint8,
    unpack_ternary_indices_from_uint8,
)

LEVELS_2B = (-1.510, -0.453, 0.453, 1.510)  # Lloyd-Max Gaussian 4-level, units of sigma
TERN_THRESH = 0.612                          # Lloyd-Max Gaussian ternary threshold, units of sigma

CONFIGS = {
    # name: (kron_rank, residual_type)
    "kron13_none": (13, "none"),
    "nokron_1bit": (0, "1bit"),
    "kron13_1bit": (13, "1bit"),
    "nokron_tern": (0, "tern"),
    "kron9_tern": (9, "tern"),
    "nokron_2bit": (0, "2bit"),
}


# ==============================================================================
# Expert bank: W_e = (Kron_e + g ⊙ Rhat_e) * exp(log_r) * exp(log_c)
# ==============================================================================
class ExpertBank(nn.Module):
    def __init__(self, E, d_out, d_in, rank, res, p1=32, q1=32):
        super().__init__()
        assert rank > 0 or res != "none", "need at least one of Kronecker core or residual"
        self.E, self.d_out, self.d_in, self.rank, self.res = E, d_out, d_in, rank, res
        self.p1, self.q1, self.p2, self.q2 = p1, q1, d_out // p1, d_in // q1
        n = E * d_out * d_in

        # Registered here (not in init_from_teacher) so a fresh module has identical keys for strict reload.
        self.log_r = nn.Parameter(torch.zeros(E, d_out, 1, dtype=torch.float16))
        self.log_c = nn.Parameter(torch.zeros(E, 1, d_in, dtype=torch.float16))
        if rank > 0:
            self.A = nn.Parameter(torch.zeros(E, rank, p1, q1, dtype=torch.float16))
            self.B = nn.Parameter(torch.zeros(E, rank, self.p2, self.q2, dtype=torch.float16))
        if res != "none":
            nbytes = {"1bit": n // 8, "2bit": n // 4, "tern": math.ceil(n / 5)}[res]
            self.register_buffer("packed", torch.zeros(nbytes, dtype=torch.uint8))
            self.gain = nn.Parameter(torch.ones(E, d_out, 1, dtype=torch.float16))
        self._rhat = None  # transient decoded residual; rebuilt from `packed`, never serialized

    def _dev(self):
        return self.log_r.device

    def kron(self, dtype):
        if self.rank == 0:
            return None
        return torch.einsum("erij,erkl->eikjl", self.A.to(dtype), self.B.to(dtype)).reshape(
            self.E, self.d_out, self.d_in
        )

    def _codes(self):
        shape = (self.E, self.d_out, self.d_in)
        if self.res == "1bit":
            return unpack_1bit_signs_from_uint8(self.packed, shape, torch.float32)
        if self.res == "tern":
            return unpack_ternary_indices_from_uint8(self.packed, shape).float() - 1.0
        lv = torch.tensor(LEVELS_2B, device=self._dev())
        return lv[unpack_2bit_indices_from_uint8(self.packed, shape)]

    @torch.no_grad()
    def rhat(self):
        if self._rhat is None:
            Ho = get_hadamard_matrix(self.d_out, self._dev())
            Hi = get_hadamard_matrix(self.d_in, self._dev())
            # Sylvester H/sqrt(n) is symmetric and self-inverse: rotate and un-rotate are the same op.
            self._rhat = (Ho @ self._codes() @ Hi).half()
        return self._rhat

    def synthesize(self, dtype=torch.float32):
        K = self.kron(dtype)
        W = K if K is not None else 0.0
        if self.res != "none":
            W = W + self.gain.to(dtype) * self.rhat().to(dtype)
        return W * torch.exp(self.log_r.to(dtype)) * torch.exp(self.log_c.to(dtype))

    @torch.no_grad()
    def init_from_teacher(self, W, act_rms):
        E = self.E
        a = act_rms.reshape(-1, 1, self.d_in).float() + 1e-6
        W_act = W.float() * a
        r = W_act.pow(2).mean(2, keepdim=True).sqrt() + 1e-8
        c = (W_act / r).pow(2).mean(1, keepdim=True).sqrt() + 1e-8
        W_bal = W_act / (r * c)
        self.log_r.copy_(torch.log(r).half())
        self.log_c.copy_(torch.log(c / a).half())

        if self.rank > 0:
            K = (
                W_bal.reshape(E, self.p1, self.p2, self.q1, self.q2)
                .permute(0, 1, 3, 2, 4)
                .reshape(E, self.p1 * self.q1, self.p2 * self.q2)
            )
            U, S, Vh = torch.linalg.svd(K, full_matrices=False)
            s = S[:, : self.rank].sqrt()
            A = (U[:, :, : self.rank] * s[:, None, :]).permute(0, 2, 1).reshape(E, self.rank, self.p1, self.q1)
            B = (Vh[:, : self.rank, :] * s[:, :, None]).reshape(E, self.rank, self.p2, self.q2)
            self.A.copy_(A.half())
            self.B.copy_(B.half())

        self.repack(W)

    @torch.no_grad()
    def repack(self, W):
        """Re-quantize R = W/scale - Kron against the CURRENT core and scales."""
        if self.res == "none":
            return
        dev = W.device
        scale = torch.exp(self.log_r.float()) * torch.exp(self.log_c.float())  # strictly positive
        target = W.float() / scale
        K = self.kron(torch.float32)
        res = target - K if K is not None else target

        Ho = get_hadamard_matrix(self.d_out, dev)
        Hi = get_hadamard_matrix(self.d_in, dev)
        x = Ho @ res @ Hi
        sig = x.std(dim=(1, 2), keepdim=True) + 1e-12  # per-expert sigma: encode/decode consistent

        if self.res == "1bit":
            q = torch.where(x >= 0, 1.0, -1.0)
            self.packed.copy_(pack_1bit_signs_to_uint8(q))
        elif self.res == "tern":
            t = TERN_THRESH * sig
            q = torch.where(x > t, 1.0, torch.where(x < -t, -1.0, 0.0))
            self.packed.copy_(pack_ternary_indices_to_uint8((q + 1.0).long()))
        else:
            lv = torch.tensor(LEVELS_2B, device=dev)
            idx = torch.argmin(((x / sig).unsqueeze(-1) - lv).abs(), dim=-1)
            self.packed.copy_(pack_2bit_indices_to_uint8(idx))
        del x

        # LS gain per UNROTATED row, i.e. exactly how it's applied in synthesize().
        self._rhat = None
        R = self.rhat().float()
        g = (res * R).sum(2, keepdim=True) / (R.pow(2).sum(2, keepdim=True) + 1e-12)
        self.gain.copy_(g.to(self.gain.dtype))


# ==============================================================================
# Routed SwiGLU MoE block
# ==============================================================================
class Block(nn.Module):
    def __init__(self, E, d_model, d_ffn, top_k, rank, res):
        super().__init__()
        self.E, self.d_model, self.top_k = E, d_model, top_k
        self.register_buffer("router_w", torch.zeros(E, d_model, dtype=torch.float16))
        self.out_gain = nn.Parameter(torch.ones(1, dtype=torch.float16))
        self.gate = ExpertBank(E, d_ffn, d_model, rank, res)
        self.up = ExpertBank(E, d_ffn, d_model, rank, res)
        self.down = ExpertBank(E, d_model, d_ffn, rank, res)
        self._return_tuple = False

    def banks(self):
        return (self.gate, self.up, self.down)

    def synth_all(self, dtype):
        return tuple(b.synthesize(dtype) for b in self.banks())

    def forward_moe(self, x, Ws=None, router_x=None):
        dt = x.dtype
        Wg, Wu, Wd = Ws if Ws is not None else self.synth_all(dt)
        logits = F.linear(router_x if router_x is not None else x, self.router_w.to(dt))
        probs = F.softmax(logits.float(), dim=-1)
        tp, ti = torch.topk(probs, self.top_k, dim=-1)
        tp = (tp / (tp.sum(-1, keepdim=True) + 1e-9)).to(dt)
        out = torch.zeros_like(x)
        for e in range(self.E):
            tok, kp = (ti == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = x.index_select(0, tok)
            h = F.silu(F.linear(xe, Wg[e])) * F.linear(xe, Wu[e])
            out.index_add_(0, tok, tp[tok, kp].unsqueeze(-1) * F.linear(h, Wd[e]))
        return out * self.out_gain.to(dt)

    def forward(self, hidden_states, *args, **kwargs):
        shp = hidden_states.shape
        y = self.forward_moe(hidden_states.reshape(-1, self.d_model)).reshape(shp)
        return (y, None) if self._return_tuple else y

    @torch.no_grad()
    def init_from_teacher(self, router_w, w_in, w_out, X, Y, dev):
        self.router_w.copy_(router_w.half().to(dev))
        wg, wu = w_in.to(dev).chunk(2, dim=1)
        wd = w_out.to(dev)
        x = X[:4096].to(dev).float()
        y = Y[:4096].to(dev).float()
        rms_in = x.pow(2).mean(0).sqrt()

        _, ti = torch.topk(F.linear(x, self.router_w.float()), self.top_k, dim=-1)
        mids = []
        for e in range(self.E):
            tok = (ti == e).any(-1).nonzero(as_tuple=True)[0]
            xe = x[tok] if tok.numel() > 16 else x[:256]
            h = F.silu(F.linear(xe, wg[e])) * F.linear(xe, wu[e])
            mids.append(h.pow(2).mean(0).sqrt())
        rms_mid = torch.stack(mids)

        self.gate.init_from_teacher(wg, rms_in)
        self.up.init_from_teacher(wu, rms_in)
        self.down.init_from_teacher(wd, rms_mid)

        self.out_gain.fill_(1.0)
        yp = self.forward_moe(x)
        alpha = float((yp * y).sum() / (yp.pow(2).sum() + 1e-12))
        self.out_gain.fill_(min(max(alpha, 0.05), 20.0))


# ==============================================================================
# Eval / storage
# ==============================================================================
@torch.no_grad()
def eval_block(blk, X, Y, dev, chunk=2048):
    blk.eval()
    P = torch.cat([blk.forward_moe(X[i : i + chunk].to(dev).float()).cpu() for i in range(0, X.shape[0], chunk)])
    P, T = P.float(), Y.float()
    mse = (P - T).pow(2).mean().item()
    return {
        "r2_pct": round(100.0 * (1.0 - mse / (T.var().item() + 1e-12)), 2),
        "cos": round(F.cosine_similarity(P.flatten(), T.flatten(), dim=0).item(), 5),
    }


def exact_bpw(blk):
    n = blk.E * 3 * blk.gate.d_out * blk.gate.d_in
    b = sum(v.numel() * v.element_size() for k, v in blk.state_dict().items() if k != "router_w")
    return round(8.0 * b / n, 4), int(b)


# ==============================================================================
# Per-layer distillation
# ==============================================================================
def train_layer(name, rank, res, t, X_cal_in, X_ev_in, dev, args, ckpt):
    E, two_f, d = t["w_in"].shape
    mk = lambda: Block(E, d, two_f // 2, t["top_k"], rank, res).to(dev)

    blk = mk()
    blk._return_tuple = t["returns_tuple"]
    blk.init_from_teacher(t["router_w"], t["w_in"], t["w_out"], t["X_calib_clean"], t["Y_calib_teacher"], dev)
    oneshot = eval_block(blk, X_ev_in, t["Y_eval_teacher"], dev)

    wg, wu = t["w_in"].to(dev).chunk(2, dim=1)
    teach = (wg, wu, t["w_out"].to(dev))
    teach_pow = [T.pow(2).mean() for T in teach]

    blk.train()
    params = [p for p in blk.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    scale_ids = {id(b.log_r) for b in blk.banks()} | {id(b.log_c) for b in blk.banks()}
    scale_p = [p for p in params if id(p) in scale_ids]
    other_p = [p for p in params if id(p) not in scale_ids]
    opt = torch.optim.AdamW(
        [{"params": other_p, "lr": args.lr}, {"params": scale_p, "lr": args.lr_scale}],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    steps = args.steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )

    X_clean, Y = t["X_calib_clean"], t["Y_calib_teacher"]
    N = X_cal_in.shape[0]
    var = Y.float().var().item() + 1e-12
    gen = torch.Generator().manual_seed(42)
    repack_at = {int(0.30 * steps), int(0.60 * steps)}
    t0 = time.perf_counter()
    act_loss_val = float("nan")

    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (args.batch_tokens,), generator=gen)
        x = X_cal_in[idx].to(dev).float()
        xc = X_clean[idx].to(dev).float()
        y = Y[idx].to(dev).float()

        Ws = blk.synth_all(torch.float32)
        router_x = xc if step < int(0.65 * steps) else None  # teacher-forced routing, then live routing
        yp = blk.forward_moe(x, Ws, router_x=router_x)
        act = F.mse_loss(yp, y) / var
        wl = sum(((W - T).pow(2).mean() / P) for W, T, P in zip(Ws, teach, teach_pow))
        loss = act + args.anchor * wl
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{name}] non-finite loss at step {step}: act={act.item()} w={wl.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        act_loss_val = act.item()

        if res != "none" and step in repack_at:
            for b, T in zip(blk.banks(), teach):
                b.repack(T)
                opt.state.pop(b.gain, None)  # stale moments belong to the pre-repack gain

    with torch.no_grad():
        for p in params:
            p.data = p.data.half()
    dt = time.perf_counter() - t0

    torch.save(blk.state_dict(), ckpt)
    re = mk()
    re._return_tuple = t["returns_tuple"]
    re.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True), strict=True)
    re.eval()

    bpw, nbytes = exact_bpw(re)
    closed = eval_block(re, X_ev_in, t["Y_eval_teacher"], dev)
    opened = eval_block(re, t["X_eval_clean"], t["Y_eval_teacher"], dev)
    return re, {
        "bpw": bpw,
        "bytes": nbytes,
        "oneshot_r2": oneshot["r2_pct"],
        "closed_r2": closed["r2_pct"],
        "closed_cos": closed["cos"],
        "open_r2": opened["r2_pct"],
        "open_cos": opened["cos"],
        "final_calib_nmse": round(act_loss_val, 5),
        "train_sec": round(dt, 1),
    }


# ==============================================================================
# Main
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12, 16, 20])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()), choices=list(CONFIGS.keys()))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1e-3, help="LR for log-scales (relative step size)")
    ap.add_argument("--anchor", type=float, default=0.05, help="weight of normalized weight-space anchor loss")
    ap.add_argument("--batch_tokens", type=int, default=1024)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--out_dir", default="./rpm_ihog_kron_ablation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.out_dir
    )
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
        rw, wi, wo, tk = extract_granite_moe_weights(orig[l])
        teacher[l] = dict(
            X_calib_clean=Xc, Y_calib_teacher=Yc, X_eval_clean=Xe, Y_eval_teacher=Ye,
            returns_tuple=rt, router_w=rw, w_in=wi, w_out=wo, top_k=tk,
        )

    results = {"baseline_ppl": base["heldout_ppl"], "args": vars(args), "configs": {}}
    for name in args.configs:
        rank, res = CONFIGS[name]
        print(f"\n=== {name} (rank={rank}, residual={res}) ===")
        for l in args.layers:
            model.model.layers[l].block_sparse_moe = orig[l]

        per_layer = {}
        for i, l in enumerate(args.layers):
            t = teacher[l]
            if i == 0:
                Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
            else:  # closed loop: inputs produced by already-compressed earlier layers
                Xci, _, _ = capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
                Xei, _, _ = capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
            blk, m = train_layer(name, rank, res, t, Xci, Xei, dev, args,
                                 os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"))
            model.model.layers[l].block_sparse_moe = blk
            per_layer[l] = m
            print(f"  L{l:2d} bpw={m['bpw']:.4f} oneshot={m['oneshot_r2']:6.2f}%  "
                  f"closed={m['closed_r2']:6.2f}%  open={m['open_r2']:6.2f}%  ({m['train_sec']}s)")

        fm, _ = evaluate_full_model_heldout(model, evalt, dev, baseline_top1=base_top1)
        avg = lambda k: round(sum(m[k] for m in per_layer.values()) / len(per_layer), 2)
        results["configs"][name] = {
            "rank": rank, "res": res, "bpw": per_layer[args.layers[0]]["bpw"],
            "avg_closed_r2": avg("closed_r2"), "avg_open_r2": avg("open_r2"),
            "ppl": fm["heldout_ppl"], "ppl_delta": round(fm["heldout_ppl"] - base["heldout_ppl"], 3),
            "top1": fm["heldout_top1_agreement_pct"], "per_layer": per_layer,
        }
        print(f"  -> PPL {fm['heldout_ppl']} (Δ {results['configs'][name]['ppl_delta']:+}) "
              f"top1 {fm['heldout_top1_agreement_pct']}%")
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    for l in args.layers:
        model.model.layers[l].block_sparse_moe = orig[l]

    print(f"\n{'config':<14}{'bpw':>8}{'closedR2':>10}{'openR2':>9}{'PPL':>9}{'ΔPPL':>9}{'top1':>8}")
    for name, r in sorted(results["configs"].items(), key=lambda kv: kv[1]["bpw"]):
        print(f"{name:<14}{r['bpw']:>8.4f}{r['avg_closed_r2']:>10.2f}{r['avg_open_r2']:>9.2f}"
              f"{r['ppl']:>9.3f}{r['ppl_delta']:>+9.3f}{r['top1']:>8.2f}")


if __name__ == "__main__":
    main()
