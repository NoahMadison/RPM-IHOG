#!/usr/bin/env python3
r"""
rpm_ihog_round6_probe.py — does GPTQ rounding of the residual generalise, and which damping?
============================================================================================
One-shot (no distillation) screen on a few layers with CLEAN teacher inputs. Calibration-internal
split only: Hessian + init are fitted on calib sequences [0, n_fit), scored on [n_fit, 48).
The held-out evaluation text is never touched, so --damp chosen here is not selected on held-out.

Score per (layer, residual type, variant):
  rel_r2    = 100 * (1 - sink-proof per-token-relative MSE)      (round-5 objective, score split)
  r2_nopos0 = plain output R^2 excluding position 0               (score split)
"""

import argparse
import json
import os

import torch

import rpm_ihog_15bpw_hadamard_distill as V1
import rpm_ihog_round4 as R4
import rpm_ihog_round5 as R5
import rpm_ihog_round6 as R6


@torch.no_grad()
def score(blk, X, Y, seq_len):
    P = torch.cat([blk.forward_moe(X[i : i + 2048].float()) for i in range(0, X.shape[0], 2048)])
    Yf = Y.float()
    keep = (torch.arange(X.shape[0], device=X.device) % seq_len) != 0
    return {"rel_r2": 100.0 * (1.0 - float(R5.rel_mse(P, Yf))),
            "r2_nopos0": 100.0 * (1.0 - float((P[keep] - Yf[keep]).pow(2).mean() / Yf[keep].var()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
    ap.add_argument("--layers", type=int, nargs="+", default=[3, 12, 28])
    ap.add_argument("--res", nargs="+", default=["1bit", "2bit", "2b1b"])
    ap.add_argument("--damps", type=float, nargs="+", default=[0.01, 0.05, 0.2, 1.0])
    ap.add_argument("--n_fit", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} exists; pick a fresh --out_dir")
    os.makedirs(args.out_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    orig = {l: R4.get_ffn(model, l) for l in range(len(model.model.layers))}
    cut = args.n_fit * args.seq_len
    out = {"args": vars(args), "results": {}}
    for l in args.layers:
        t = R4.teacher_for(model, orig, [], l, calib, calib[:1], dev)  # eval side unused
        Xf, Yf = t["X_calib_clean"][:cut], t["Y_calib_teacher"][:cut]
        Xs, Ys = t["X_calib_clean"][cut:], t["Y_calib_teacher"][cut:]
        E, two_f, d = t["w_in"].shape
        Cin, Cmid, ntok = R6.expert_hessians(t["router_w"], t["w_in"], t["top_k"], Xf, Yf)
        out["results"][str(l)] = {"hess_tokens_min": min(ntok), "hess_tokens_median": sorted(ntok)[len(ntok) // 2]}
        for r in args.res:
            cls = R6.mix_cls6(r, r, r)
            row = {}
            for damp in [None] + args.damps:
                R6.GPTQ["damp"] = damp or 0.0
                R6.CTX["hess"] = None if damp is None else (Cin, Cmid)
                blk = cls(E, d, two_f // 2, t["top_k"], 13, "mix").to(dev)
                blk._return_tuple = t["returns_tuple"]
                blk.init_from_teacher(t["router_w"], t["w_in"], t["w_out"], Xf, Yf, dev)
                R5.refit_gain_(blk, Xf, Yf)
                s = score(blk, Xs, Ys, args.seq_len)
                key = "nearest" if damp is None else f"gptq_d{damp}"
                row[key] = s
                print(f"  L{l:2d} {r:<5} {key:<12} rel_r2={s['rel_r2']:6.2f}  r2_nopos0={s['r2_nopos0']:6.2f}", flush=True)
                del blk
                torch.cuda.empty_cache()
            R6.CTX["hess"] = None
            out["results"][str(l)][r] = row
        del t, Cin, Cmid
        torch.cuda.empty_cache()
        with open(os.path.join(args.out_dir, "probe.json"), "w") as f:
            json.dump(out, f, indent=1)
    print(f"[+] {os.path.join(args.out_dir, 'probe.json')}")


if __name__ == "__main__":
    main()
