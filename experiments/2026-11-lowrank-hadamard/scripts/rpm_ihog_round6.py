#!/usr/bin/env python3
r"""
rpm_ihog_round6.py — Hessian-aware (GPTQ/LDLQ) rounding of the Hadamard-domain residual codes
=============================================================================================
What changed vs round 5, and why
  Round 5 rounds every rotated residual coefficient to its NEAREST Lloyd-Max level, independently.
  Weight-space R² then sits exactly at the scalar Lloyd-Max ceiling, but weight-space error is the
  wrong target: what matters is the OUTPUT error  E_x |dW x|^2 = tr(dW C dW^T),  C = E[x x^T].
  GPTQ/LDLQ minimises that directly: quantise one input column at a time and push its rounding error
  onto the not-yet-quantised columns along C^-1. Storage is unchanged (same codes, same gains), so
  bpw is identical to round 5; only which code is chosen per coefficient changes.

  Where it is applied. Each bank stores  W = (K + g . H_o Q H_i) * r * c  (row scales r, column
  scales c, per-row gain g). With z = c . x the residual's contribution to the output error is
       sum_i (g_i r_i)^2 [H_o E H_i z]_i^2,   E = Q*s - X (rotated domain).
  Treating g.r as constant across rows (H_o mixes rows anyway) this is tr(E Hrot E^T) with
       Hrot = H_i (c C c) H_i,
  i.e. ordinary row-separable GPTQ in the rotated domain with ONE Hessian per expert. The row gains
  are then refitted by C-weighted least squares (2x2 normal equations for the 2-bit+1-bit stack).

  C per expert: routed calibration inputs (the DRIFTED closed-loop inputs the block actually sees),
  each token weighted by (router prob)^2 x the round-5 sink-proof weight 1/(|y_t|^2 + median|y|^2),
  so the Hessian sees the same objective the distillation optimises. For `down` the input is the
  teacher's h = silu(x Wg^T) * (x Wu^T) on those same inputs. Relative damping --damp (picked on a
  calibration-internal split by rpm_ihog_round6_probe.py, never on held-out text).

  Every repack (init, 30 %, 60 % of training) uses GPTQ; the final 40 % of steps train only the
  continuous parameters (factors, scales, gains) with the codes fixed.

Configs added (suffix _gptq = same storage as the round-5 config of the same name):
  kron13_1bit_gptq, kron13_2bit_gptq, kron13_2b1b_gptq, gate2_1bit_gptq

  CUDA_VISIBLE_DEVICES=1 python rpm_ihog_round6.py --model_id ibm-granite/granite-3.1-3b-a800m-base \
      --configs kron13_2bit_gptq kron13_2b1b_gptq --out_dir ./rpm_ihog_round6_3b_<tag>
"""

import argparse
import os

import torch
import torch.nn.functional as F

import rpm_ihog_15bpw_hadamard_distill as V1
import rpm_ihog_kron_ablation as KA
import rpm_ihog_round3 as R3
import rpm_ihog_round4 as R4
import rpm_ihog_round5 as R5

LV2 = getattr(KA, "LEVELS_2B", (-1.510, -0.453, 0.453, 1.510))
GPTQ = {"damp": 0.05, "block": 128}      # module-level so banks created inside R5.train_layer5 see it
CTX = {"hess": None}                     # (C_in, C_mid) for the layer currently being built


# ------------------------------------------------------------------------------
# Per-expert input covariances (sink-proof token weights x router prob^2)
# ------------------------------------------------------------------------------
@torch.no_grad()
def expert_hessians(router_w, w_in, top_k, X, Y, chunk=4096):
    x = X.float()
    E, two_f, d = w_in.shape
    f = two_f // 2
    e_t = Y.float().pow(2).sum(1)
    wt = 1.0 / (e_t + e_t.median() + 1e-12)
    wt = wt / wt.mean()
    if E == 1:
        ti = torch.zeros(x.shape[0], 1, dtype=torch.long, device=x.device)
        tp = torch.ones(x.shape[0], 1, device=x.device)
    else:
        probs = F.softmax(F.linear(x, router_w.float()), dim=-1)
        tp, ti = torch.topk(probs, top_k, dim=-1)
        tp = tp / tp.sum(-1, keepdim=True)
    Cin = torch.zeros(E, d, d, device=x.device)
    Cmid = torch.zeros(E, f, f, device=x.device)
    ntok = []
    for e in range(E):
        tok, kp = (ti == e).nonzero(as_tuple=True)
        ntok.append(int(tok.numel()))
        if tok.numel() == 0:
            Cin[e] = torch.eye(d, device=x.device)
            Cmid[e] = torch.eye(f, device=x.device)
            continue
        a = tp[tok, kp].pow(2) * wt[tok]
        a = a / a.sum()
        for i in range(0, tok.numel(), chunk):
            xe, ae = x[tok[i : i + chunk]], a[i : i + chunk, None]
            h = F.silu(xe @ w_in[e, :f].float().T) * (xe @ w_in[e, f:].float().T)
            Cin[e] += (xe * ae).T @ xe
            Cmid[e] += (h * ae).T @ h
    return Cin, Cmid, ntok


# ------------------------------------------------------------------------------
# Batched GPTQ / LDLQ in the rotated domain
# ------------------------------------------------------------------------------
@torch.no_grad()
def _inv_chol_upper(H, damp):
    E, C, _ = H.shape
    d = H.diagonal(dim1=1, dim2=2)
    I = torch.eye(C, device=H.device)
    lam = damp * d.mean(-1).clamp_min(1e-12)
    for _ in range(6):
        L, info = torch.linalg.cholesky_ex(H + lam[:, None, None] * I)
        if int(info.max()) == 0:
            break
        lam = torch.where(info > 0, lam * 10, lam)
    Hinv = torch.cholesky_inverse(L)
    U, info = torch.linalg.cholesky_ex(Hinv, upper=True)
    if int(info.max()) != 0:
        raise RuntimeError("GPTQ: inverse Cholesky failed")
    return U


@torch.no_grad()
def gptq(X, Hrot, quant, damp, block=128):
    """X: (E, R, C) rotated residual; Hrot: (E, C, C). quant(w (E,R), j) -> q (E,R), records codes."""
    E, R, C = X.shape
    U = _inv_chol_upper(Hrot, damp)
    W = X.clone()
    for i1 in range(0, C, block):
        i2 = min(i1 + block, C)
        W1 = W[:, :, i1:i2].clone()
        Err = torch.zeros_like(W1)
        U1 = U[:, i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, :, i]
            q = quant(w, i1 + i)
            err = (w - q) / U1[:, i, i][:, None]
            W1[:, :, i:] -= err[:, :, None] * U1[:, i, i:][:, None, :]
            Err[:, :, i] = err
        if i2 < C:
            W[:, :, i2:] -= Err @ U[:, i1:i2, i2:]
    return W


@torch.no_grad()
def gptq_repack(bank, W):
    """Drop-in for ExpertBank.repack / Bank21.repack using bank._hess (unrotated input covariance)."""
    dev = W.device
    res_t = bank.res if bank.res in ("1bit", "2bit", "2b1b") else None
    if res_t is None:
        raise ValueError(f"gptq_repack: unsupported residual {bank.res}")
    scale = torch.exp(bank.log_r.float()) * torch.exp(bank.log_c.float())
    K = bank.kron(torch.float32)
    res = W.float() / scale - (K if K is not None else 0.0)
    Ho, Hi = R4.block_hadamard(bank.d_out, dev), R4.block_hadamard(bank.d_in, dev)
    X = Ho @ res @ Hi
    c = torch.exp(bank.log_c.float())                         # (E, 1, d_in)
    Cz = bank._hess * c.transpose(1, 2) * c
    Hrot = Hi @ Cz @ Hi
    E, Rr, Cc = X.shape
    lv = torch.tensor(LV2, device=dev)
    sig = X.std(dim=(1, 2)) + 1e-12                           # (E,)
    if res_t == "1bit":
        s = X.abs().mean(dim=(1, 2))
        S = torch.empty_like(X)

        def quant(w, j):
            sg = torch.where(w >= 0, 1.0, -1.0)
            S[:, :, j] = sg
            return sg * s[:, None]
    else:
        I2 = torch.empty(X.shape, dtype=torch.long, device=dev)
        if res_t == "2b1b":
            n_idx = torch.argmin(((X / sig[:, None, None]).unsqueeze(-1) - lv).abs(), dim=-1)
            s1 = (X - lv[n_idx] * sig[:, None, None]).abs().mean(dim=(1, 2))
            del n_idx
            S = torch.empty_like(X)

        def quant(w, j):
            idx = torch.argmin(((w / sig[:, None]).unsqueeze(-1) - lv).abs(), dim=-1)
            I2[:, :, j] = idx
            q = lv[idx] * sig[:, None]
            if res_t == "2b1b":
                sg = torch.where(w - q >= 0, 1.0, -1.0)
                S[:, :, j] = sg
                q = q + sg * s1[:, None]
            return q

    gptq(X, Hrot, quant, GPTQ["damp"], GPTQ["block"])
    del X, Hrot
    if res_t == "1bit":
        bank.packed.copy_(V1.pack_1bit_signs_to_uint8(S))
    else:
        bank.packed.copy_(V1.pack_2bit_indices_to_uint8(I2))
        if res_t == "2b1b":
            bank.packed1.copy_(V1.pack_1bit_signs_to_uint8(S))
    bank._rhat = None
    if res_t == "2b1b":
        bank._rhat1 = None
        R2, R1 = bank.rhat().float(), bank.rhat1().float()
        R2C, R1C = R2 @ Cz, R1 @ Cz
        a, b, cc = (R2 * R2C).sum(2), (R2 * R1C).sum(2), (R1 * R1C).sum(2)
        u, v = (res * R2C).sum(2), (res * R1C).sum(2)
        det = (a * cc - b * b).clamp_min(1e-12)
        bank.gain.copy_(((cc * u - b * v) / det).unsqueeze(-1).to(bank.gain.dtype))
        bank.gain1.copy_(((a * v - b * u) / det).unsqueeze(-1).to(bank.gain1.dtype))
    else:
        R = bank.rhat().float()
        RC = R @ Cz
        g = (res * RC).sum(2, keepdim=True) / ((R * RC).sum(2, keepdim=True) + 1e-12)
        bank.gain.copy_(g.to(bank.gain.dtype))


class _GPTQMixin:
    _hess = None

    @torch.no_grad()
    def repack(self, W):
        if self._hess is None:
            return super().repack(W)
        return gptq_repack(self, W)


class GBank(_GPTQMixin, R3.Bank):
    pass


class GBank21(_GPTQMixin, R4.Bank21):
    pass


_MIX6 = {}


def mix_cls6(gate, up, down):
    key = (gate, up, down)
    if key in _MIX6:
        return _MIX6[key]

    class BlockMix6(R3.Block3):
        def __init__(self, E, d_model, d_ffn, top_k, rank, res=None, whiten=False, expalloc=False, gamma=0.5):
            super().__init__(E, d_model, d_ffn, top_k, rank, "1bit", whiten, expalloc, gamma)

            def mk(r, do, di):
                return GBank21(E, do, di, rank) if r == "2b1b" else GBank(E, do, di, rank, r)

            self.gate = mk(gate, d_ffn, d_model)
            self.up = mk(up, d_ffn, d_model)
            self.down = mk(down, d_model, d_ffn)

        @torch.no_grad()
        def init_from_teacher(self, router_w, w_in, w_out, X, Y, dev):
            if CTX["hess"] is not None:
                Cin, Cmid = CTX["hess"]
                self.gate._hess, self.up._hess, self.down._hess = Cin, Cin, Cmid
            return super().init_from_teacher(router_w, w_in, w_out, X, Y, dev)

    BlockMix6.__name__ = BlockMix6.__qualname__ = f"BlockMix6_{gate}_{up}_{down}"
    _MIX6[key] = BlockMix6
    return BlockMix6


def _cfg6(tag, g, u, d, rank=13):
    return {"tag": tag, "rank": rank, "res": "mix", "block_cls": mix_cls6(g, u, d), "gptq": True,
            "residuals": {"gate": g, "up": u, "down": d}}


CONFIGS6 = {
    "kron13_1bit_gptq": _cfg6("kron13_1bit_gptq", "1bit", "1bit", "1bit"),
    "gate2_1bit_gptq": _cfg6("gate2_1bit_gptq", "2bit", "1bit", "1bit"),
    "kron13_2bit_gptq": _cfg6("kron13_2bit_gptq", "2bit", "2bit", "2bit"),
    "kron13_2b1b_gptq": _cfg6("kron13_2b1b_gptq", "2b1b", "2b1b", "2b1b"),
}
ALL_CONFIGS = {**R5.CONFIGS5, **CONFIGS6}


# ------------------------------------------------------------------------------
# Chain: same as round 5, plus Hessians from the drifted calibration inputs
# ------------------------------------------------------------------------------
def run_chain6(name, spec, model, orig, layers, calib, evalt, dev, args):
    replaced, per = [], {}
    for l in layers:
        t = R4.teacher_for(model, orig, replaced, l, calib, evalt, dev)
        if replaced:
            Xci, _, _ = R3.capture_real_moe_io(model, orig[l], calib, dev, capture_outputs=False)
            Xei, _, _ = R3.capture_real_moe_io(model, orig[l], evalt, dev, capture_outputs=False)
        else:
            Xci, Xei = t["X_calib_clean"], t["X_eval_clean"]
        ntok = None
        if spec.get("gptq"):
            Cin, Cmid, ntok = expert_hessians(t["router_w"], t["w_in"], t["top_k"], Xci, t["Y_calib_teacher"])
            CTX["hess"] = (Cin, Cmid)
        try:
            blk, m = R5.train_layer5(name, spec, t, Xci, Xei, dev, args,
                                     os.path.join(args.out_dir, f"ckpt_{name}_L{l}.pt"))
        finally:
            CTX["hess"] = None
        if ntok is not None:
            m["hess_tokens_min"], m["hess_tokens_median"] = min(ntok), sorted(ntok)[len(ntok) // 2]
        R4.set_ffn(model, l, blk)
        replaced.append(l)
        per[l] = m
        print(f"  L{l:2d} bpw={m['bpw']:.3f} 1shot open/closed={m['oneshot_open']:6.1f}/{m['oneshot_closed']:6.1f}  "
              f"open={m['open_r2']:6.1f} (no tok0 {m['open_r2_nopos0']:5.1f})  closed={m['closed_r2']:6.1f}  "
              f"gain={m['out_gain']:.2f}  tok0={m['y_pos0']:.0%}  wR2 g/u/d={m['wR2_gate']:.0f}/{m['wR2_up']:.0f}/"
              f"{m['wR2_down']:.0f}  ({m['train_sec']}s)", flush=True)
        del t, Xci, Xei
        CTX["hess"] = None
        torch.cuda.empty_cache()
    return per


def install():
    """Route round-5's driver through the round-6 configs/chain (in-memory only; no file is touched)."""
    R5.CONFIGS5.update(CONFIGS6)
    R5.run_chain5 = run_chain6


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
    ap.add_argument("--configs", nargs="+", default=["kron13_2bit_gptq", "kron13_2b1b_gptq"], choices=list(ALL_CONFIGS))
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
    ap.add_argument("--damp", type=float, default=GPTQ["damp"])
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", required=True)
    return ap


def main():
    args = build_parser().parse_args()
    if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
        raise SystemExit(f"{args.out_dir} exists and is not empty; pick a fresh --out_dir")
    GPTQ["damp"] = args.damp
    install()
    print("[round 6] vs round 5: residual codes chosen by GPTQ/LDLQ in the Hadamard domain "
          f"(per-expert sink-weighted input Hessian, damp={args.damp}); storage/bpw unchanged.")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    R5.run_round5(model, calib, evalt, args, dev)


if __name__ == "__main__":
    main()
