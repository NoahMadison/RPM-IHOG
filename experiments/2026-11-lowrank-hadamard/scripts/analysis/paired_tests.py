#!/usr/bin/env python3
"""Paired bootstrap of PPL(row) - PPL(reference) over the same 24 eval sequences, from per-sequence NLLs stored in
results/round8_v3_results.json and results/round8_v4_results.json (no model, no GPU). Writes results/paired_tests.json.

    python scripts/analysis/paired_tests.py   # run from experiments/2026-11-lowrank-hadamard/

Recipe: torch.Generator seed 0, 10000 resamples of 24 sequence indices with replacement, 2.5/97.5 % quantiles.
Positive = row has higher perplexity (worse) than the reference. Covers eval-sampling noise only, not run-to-run
training noise.
"""
import json
import os

import torch

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results")


def rows_of(path):
    with open(os.path.join(RES, path)) as f:
        d = json.load(f)
    rows = {r["name"]: r for r in d.get("rtn", []) + d.get("controls", [])}
    for v in d["configs"].values():
        for r in v["rows"]:
            rows[r["name"]] = r
    return d, rows


def paired(a, b, n_boot=10000, seed=0):
    a, b = torch.tensor(a), torch.tensor(b)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, a.numel(), (n_boot, a.numel()), generator=g)
    d = a[idx].mean(1).exp() - b[idx].mean(1).exp()
    lo, hi = torch.quantile(d, torch.tensor([0.025, 0.975])).tolist()
    return {"dppl": round(float(a.mean().exp() - b.mean().exp()), 3), "ci95": [round(lo, 3), round(hi, 3)],
            "p_row_worse": round(float((d > 0).float().mean()), 3)}


out = {"recipe": "torch.Generator seed 0, 10000 paired resamples over 24 eval sequences; PPL(row) - PPL(ref)",
       "comparisons": []}
d3, v3 = rows_of("round8_v3_results.json")
ref = "kron13_2b1b_gptq fINT8"
for name, r in v3.items():
    if "nll_seq" in r and name != ref:
        out["comparisons"].append({"run": "round8_v3", "row": name, "ref": ref, "bpw": r["bpw"],
                                   **paired(r["nll_seq"], v3[ref]["nll_seq"])})
d4, v4 = rows_of("round8_v4_results.json")
for name in ("lr_2bit_gptq fINT8", "lr_2bit_gptq f16", "kron13_2bit_gptq_sink fINT8"):
    for refn in ("RTN INT4 g64", "RTN INT3 g64"):
        out["comparisons"].append({"run": "round8_v4", "row": name, "ref": refn, "bpw": v4[name]["bpw"],
                                   **paired(v4[name]["nll_seq"], v4[refn]["nll_seq"])})
fp = max(abs(x - y) for x, y in zip(d3["fp16_nll_seq"], d4["fp16_nll_seq"]))
out["fp16_per_seq_nll_max_abs_diff_v3_vs_v4"] = round(fp, 5)
with open(os.path.join(RES, "paired_tests.json"), "w") as f:
    json.dump(out, f, indent=1)
for c in out["comparisons"]:
    print(f"{c['run']:<10} {c['row']:<30} vs {c['ref']:<24} {c['dppl']:+.3f} {c['ci95']}  P(worse) {c['p_row_worse']}")
print("FP16 per-seq NLL max |v3 - v4|:", out["fp16_per_seq_nll_max_abs_diff_v3_vs_v4"])
