#!/usr/bin/env python3
"""Summary statistics quoted in README.md / RESULTS.md that are not single fields of a run's results file
(medians over layers, ranges over spectral summaries, replicate lists, quantizer ceilings).
Reads only results/*.json and the CEILING constant in scripts/rpm_ihog_report.py. Writes results/derived_stats.json.

    python scripts/analysis/derived_stats.py   # run from experiments/2026-11-lowrank-hadamard/
"""
import ast
import json
import os
import statistics as st

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "results")


def load(n):
    with open(os.path.join(RES, n)) as f:
        return json.load(f)


out = {}

# scalar Lloyd-Max weight-space ceilings used by the report figures (Gaussian source), and the Gaussian
# rate-distortion bound at 2 bits/weight: R^2 = 1 - 2^(-2R)
src = open(os.path.join(ROOT, "scripts", "rpm_ihog_report.py")).read()
node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CEILING")
out["lloyd_max_ceiling_wR2_pct"] = ast.literal_eval(node.value)
out["gaussian_rd_bound_2bit_wR2_pct"] = round(100 * (1 - 2 ** -4), 2)

# per-config medians over the 32 layers
per = {}
for f, cfgs in (("round5_3b_results.json", ["kron13_1bit", "gate2_1bit", "kron13_2bit", "kron13_2b1b"]),
                ("round6_3b_results.json", ["kron13_2bit_gptq", "kron13_2b1b_gptq"]),
                ("round8_v3_results.json", ["kron13_2b1b_gptq", "lr_2b1b_gptq"]),
                ("round8_v4_results.json", ["lr_2bit_gptq", "kron13_2bit_gptq_sink"])):
    d = load(f)
    for c in cfgs:
        L = d["configs"][c]["layers"]
        m = {k: round(st.median(L[l][k] for l in L), 1) for k in
             ("open_r2", "closed_r2", "oneshot_closed", "wR2_gate", "wR2_up", "wR2_down", "swap_gate", "swap_up", "swap_down")}
        per[f"{f.split('_results')[0]}:{c}"] = m
out["layer_medians"] = per

# spectral summaries (one expert per layer)
spec = {}
for tag, f in (("round5_L3", "round5_spectral_L3.json"), ("round5_L12", "round5_spectral_L12.json"),
               ("round6_L3", "round6_spectral_L3.json"), ("round6_L12", "round6_spectral_L12.json"),
               ("round7_L3", "round7_spectral_L3.json"), ("round7_L12", "round7_spectral_L12.json")):
    spec[tag] = load(f)
r8 = load("round8_spectral.json")
spec["round8_L5"], spec["round8_L12"] = r8["5"], r8["12"]
kron_t = [m["kron13_energy"]["teacher"] for s in spec.values() for m in s["matrices"].values()]
kron_g = [m["kron13_energy"]["gaussian"] for s in spec.values() for m in s["matrices"].values()]
kurt_rot = [s["residual_kurtosis"]["Hadamard-rotated residual"] for s in spec.values()]
kurt_raw = [s["residual_kurtosis"]["raw residual"] for s in spec.values()]
out["spectral"] = {
    "kron13_energy_teacher_min_max": [min(kron_t), max(kron_t)],
    "kron13_energy_gaussian_min_max": [min(kron_g), max(kron_g)],
    "residual_kurtosis_rotated_min_max": [min(kurt_rot), max(kurt_rot)],
    "residual_kurtosis_raw_min_max": [min(kurt_raw), max(kurt_raw)],
    "svd_top5pct_energy": {k: {m: v["svd_top5pct_energy"]["teacher"] for m, v in s["matrices"].items()}
                           for k, s in spec.items() if k in ("round8_L5", "round8_L12")},
    "svd_top5pct_energy_gaussian": spec["round8_L12"]["matrices"]["gate"]["svd_top5pct_energy"]["gaussian"],
}
tsnr = []
for k, s in spec.items():
    for cfg, v in s["temporal_snr_db"].items():
        tsnr.append({"spectrum": k, "config": cfg, "low": v["low_freq"], "high": v["high_freq"],
                     "high_minus_low": round(v["high_freq"] - v["low_freq"], 2)})
l12 = [t for t in tsnr if t["spectrum"].endswith("L12")]
pos = [t["high_minus_low"] for t in l12 if t["high_minus_low"] > 0]
out["temporal_snr"] = {"all": tsnr, "L12_count": len(l12), "L12_low_below_high": len(pos),
                       "L12_gap_min_max_db": [min(pos), max(pos)]}

# replicates of the same config (same code path, separate runs)
rep = {}
for fb in (16, 8, 4):
    vals = []
    for f, c in (("round6_3b_results.json", "kron13_2b1b_gptq"), ("round8_v2_results.json", "kron13_2b1b_gptq"),
                 ("round8_v3_results.json", "kron13_2b1b_gptq")):
        r = next(r for r in load(f)["configs"][c]["rows"] if r["fb"] == fb)
        vals.append(r["dppl"])
    rep[f"kron13_2b1b_gptq_f{fb}"] = {"dppl_runs_r6_r8v2_r8v3": vals, "mean": round(st.mean(vals), 3),
                                      "sd": round(st.stdev(vals), 3)}
r7 = load("round7_3b_results.json")
pre = load("round7_e2e_log.json")["pre_e2e_fp16factors"]
r6 = next(r for r in load("round6_3b_results.json")["configs"]["kron13_2bit_gptq"]["rows"] if r["fb"] == 16)
rep["kron13_2bit_gptq_f16_per_layer_stage"] = {"round6": r6["dppl"], "round7_pre_e2e": pre["dppl"]}
out["replicates"] = rep

with open(os.path.join(RES, "derived_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "temporal_snr"}, indent=1)[:4000])
print("temporal L12:", out["temporal_snr"]["L12_low_below_high"], "of", out["temporal_snr"]["L12_count"],
      "gap", out["temporal_snr"]["L12_gap_min_max_db"])
