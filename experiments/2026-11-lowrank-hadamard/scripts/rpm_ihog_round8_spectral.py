#!/usr/bin/env python3
"""Spectral analysis (rpm_ihog_spectral.analyze) on round-8 checkpoints; reads --ckpt_dir, writes only to --out_dir."""
import argparse, os, json, torch
import rpm_ihog_round8 as R8
import rpm_ihog_spectral as SP
ap = argparse.ArgumentParser()
ap.add_argument("--model_id", default="ibm-granite/granite-3.1-3b-a800m-base")
ap.add_argument("--ckpt_dir", required=True); ap.add_argument("--out_dir", required=True)
ap.add_argument("--configs", nargs="+", default=["lr_2b1b_gptq", "kron13_2b1b_gptq"])
ap.add_argument("--layers", type=int, nargs="+", default=[5, 12])
args = ap.parse_args()
if os.path.exists(args.out_dir) and os.listdir(args.out_dir):
    raise SystemExit(f"{args.out_dir} not empty")
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = torch.device("cuda")
tok = AutoTokenizer.from_pretrained(args.model_id)
_, evalt, _ = R8.R4.V1.load_disjoint_calib_and_eval_tokens(tok, 256, 48, 24, cache_dir="./rpm_ihog_15bpw_results")
model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
_, wi, _, _ = R8.R4.extract(R8.R4.get_ffn(model, 0), dev)
CONF = R8.make_configs(R8.lr_rank_matched(wi.shape[2], wi.shape[1] // 2))
out = {}
for l in args.layers:
    print(f"=== spectral L{l} {args.configs} ===", flush=True)
    out[str(l)] = SP.analyze(model, l, args.ckpt_dir, args.configs, os.path.join(args.out_dir, f"spectral_L{l}"), evalt, dev, cfg_table=CONF)
json.dump(out, open(os.path.join(args.out_dir, "spectral.json"), "w"), indent=1, default=str)
print("[spectral] done", flush=True)
