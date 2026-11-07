#!/usr/bin/env python3
r"""
rpm_ihog_eval_agree.py — post-hoc agreement metrics for saved RPM-IHOG checkpoints (read-only on all run dirs)
==============================================================================================================
Rebuilds the compressed model from per-layer checkpoints of an earlier run and scores it against the FP16
model on the SAME held-out eval split the runs used (V1.load_disjoint_calib_and_eval_tokens, same args):
  ΔPPL + bootstrap CI (sanity: must reproduce the run's own number), top-1 agreement, top-5 / top-10 set
  overlap, FP16-top-1 ∈ student-top-5, KL(p_fp16 || p_student) per token, and full-model R² of the final
  hidden state (pre-lm_head) and of the logits vs FP16. Per factor precision (16/8/4, same quantiser).
Nothing is trained or selected; writes only to --out_json.

  CUDA_VISIBLE_DEVICES=1 python rpm_ihog_eval_agree.py --out_json ./agree_<tag>.json \
      --run kron13_2bit_gptq=./rpm_ihog_round6_3b_gptq_d1_v3:ckpt_kron13_2bit_gptq_L{l}.pt ...
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

import rpm_ihog_15bpw_hadamard_distill as V1
import rpm_ihog_round3 as R3
import rpm_ihog_round4 as R4
import rpm_ihog_round6 as R6


@torch.no_grad()
def forward_all(model, tokens, dev, bs=4):
    """Returns per-batch (logits fp32 on CPU as fp16, final hidden fp16 CPU)."""
    model.eval()
    lg_all, h_all = [], []
    for i in range(0, tokens.shape[0], bs):
        b = tokens[i : i + bs].to(dev)
        out = model(b, use_cache=False, output_hidden_states=True)
        lg_all.append(out.logits[:, :-1].half().cpu())
        h_all.append(out.hidden_states[-1][:, :-1].half().cpu())
    return torch.cat(lg_all), torch.cat(h_all)


def r2(pred, ref):
    pred, ref = pred.float().flatten(0, 1), ref.float().flatten(0, 1)
    sst = (ref - ref.mean(0, keepdim=True)).pow(2).sum()
    return float(100 * (1 - (pred - ref).pow(2).sum() / sst))


@torch.no_grad()
def agreement(lg, h, ref_lg, ref_h, dev, chunk=8):
    t1 = t5 = t10 = in5 = kl = n = 0.0
    for i in range(0, lg.shape[0], chunk):
        s, r = lg[i : i + chunk].to(dev).float(), ref_lg[i : i + chunk].to(dev).float()
        ls, lr = F.log_softmax(s, -1), F.log_softmax(r, -1)
        kl += float((lr.exp() * (lr - ls)).sum(-1).sum())
        ts, tr = s.topk(10, -1).indices, r.topk(10, -1).indices
        t1 += float((ts[..., 0] == tr[..., 0]).sum())
        t5 += float((ts[..., :5, None] == tr[..., None, :5]).any(-1).float().mean(-1).sum())
        t10 += float((ts[..., :, None] == tr[..., None, :]).any(-1).float().mean(-1).sum())
        in5 += float((ts[..., :5] == tr[..., :1]).any(-1).sum())
        n += s.shape[0] * s.shape[1]
    return {"top1": round(100 * t1 / n, 2), "top5_overlap": round(100 * t5 / n, 2),
            "top10_overlap": round(100 * t10 / n, 2), "fp16top1_in_top5": round(100 * in5 / n, 2),
            "kl_per_tok": round(kl / n, 4), "r2_hidden": round(r2(h, ref_h), 2), "r2_logits": round(r2(lg, ref_lg), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
    ap.add_argument("--run", nargs="+", required=True, help="label=config:dir:pattern  (pattern has {l})")
    ap.add_argument("--factor_bits", type=int, nargs="+", default=[16, 8, 4])
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--calib_seqs", type=int, default=48)
    ap.add_argument("--eval_seqs", type=int, default=24)
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out_json):
        raise SystemExit(f"{args.out_json} exists; pick a fresh name")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    calib, evalt, _ = V1.load_disjoint_calib_and_eval_tokens(
        tok, args.seq_len, args.calib_seqs, args.eval_seqs, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    nL = len(model.model.layers)
    orig = {l: R4.get_ffn(model, l) for l in range(nL)}
    _, _, rt = R3.capture_real_moe_io(model, orig[0], evalt[:1], dev, capture_outputs=False)
    nll_fp16 = R4.seq_nll(model, evalt, dev)
    ref_lg, ref_h = forward_all(model, evalt, dev)
    out = {"model": args.model_id, "fp16_ppl": round(float(nll_fp16.mean().exp()), 3), "runs": {}}
    print(f"[agree] FP16 PPL {out['fp16_ppl']}", flush=True)

    for item in args.run:
        label, rest = item.split("=", 1)
        cfg, d, pat = rest.split(":", 2)
        spec = R6.ALL_CONFIGS[cfg]
        for l in range(nL):
            rw, wi, wo, tk = R4.extract(orig[l], dev)
            E, two_f, dm = wi.shape
            blk = spec["block_cls"](E, dm, two_f // 2, tk, spec["rank"], spec["res"]).to(dev)
            blk._return_tuple = rt
            blk.load_state_dict(torch.load(os.path.join(d, pat.format(l=l)), map_location=dev, weights_only=True),
                                strict=True)
            R4.set_ffn(model, l, blk.eval())
            del rw, wi, wo
        res = {}
        for fb in args.factor_bits:
            blocks = [R4.get_ffn(model, l) for l in range(nL)]
            saved = R4.quantize_factors_(blocks, fb) if fb < 16 else []
            nll = R4.seq_nll(model, evalt, dev)
            lg, h = forward_all(model, evalt, dev)
            res[f"f{fb}"] = {"bpw": round(sum(R4.bpw(b, fb) for b in blocks) / nL, 4), **R4.ppl_stats(nll, nll_fp16),
                             **agreement(lg, h, ref_lg, ref_h, dev)}
            R4.restore_(saved)
            print(f"[agree] {label} f{fb}: {res[f'f{fb}']}", flush=True)
            del lg, h
        out["runs"][label] = res
        for l in range(nL):
            R4.set_ffn(model, l, orig[l])
        torch.cuda.empty_cache()
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
