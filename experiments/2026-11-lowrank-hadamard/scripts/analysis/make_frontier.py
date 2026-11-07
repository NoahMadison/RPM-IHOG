#!/usr/bin/env python3
"""Plot the round-7/8 quality-vs-size frontier from the JSON files in ../../results (no model, no GPU).

    python scripts/analysis/make_frontier.py   # run from experiments/2026-11-lowrank-hadamard/
writes figures/frontier_round8.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


v3, v4, r7 = load("round8_v3_results.json"), load("round8_v4_results.json"), load("round7_3b_results.json")


def row(d, name):
    pool = d.get("rtn", []) + d.get("controls", []) + [r for v in d["configs"].values() for r in v["rows"]]
    return next(r for r in pool if r["name"] == name)


def cfg_row(d, cfg, fb):
    return next(r for r in d["configs"][cfg]["rows"] if r["fb"] == fb)


INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3 (light surface)

# (series, label, bpw, dppl, ci, colour, marker)
pts = []
for nm, lab in (("RTN INT4 g64", "RTN INT4"), ("RTN INT3 g64", "RTN INT3")):
    r = row(v3, nm); pts.append(("RTN g64 (baseline)", lab, r["bpw"], r["dppl"], r["ci95"], INK, "s"))
r = row(v3, "ctrl_gptq_int3_asym d0.01")
pts.append(("GPTQ INT3 asym g64 (baseline)", "GPTQ INT3 asym", r["bpw"], r["dppl"], r["ci95"], MUTED, "D"))
for fb, lab in ((8, "Kronecker core, 2b1b, INT8 f."), (4, "Kronecker core, 2b1b, INT4 f.")):
    r = cfg_row(v3, "kron13_2b1b_gptq", fb); pts.append(("Kronecker core (r13)", lab, r["bpw"], r["dppl"], r["ci95"], BLUE, "o"))
r = cfg_row(v3, "lr_2b1b_gptq", 8); pts.append(("low-rank core (r11)", "low-rank core, 2b1b, INT8 f.", r["bpw"], r["dppl"], r["ci95"], ORANGE, "o"))
r = cfg_row(v4, "lr_2bit_gptq", 8); pts.append(("low-rank core (r11)", "low-rank core, 2-bit + sink rule, INT8 f.", r["bpw"], r["dppl"], r["ci95"], ORANGE, "^"))
for fb, lab in ((4, "Kronecker + e2e KL, 2-bit, INT4 f."), (8, "Kronecker + e2e KL, 2-bit, INT8 f.")):
    r = cfg_row(r7, "kron13_2bit_gptq", fb); pts.append(("Kronecker + end-to-end KL (round 7)", lab, r["bpw"], r["dppl"], r["ci95"], AQUA, "^"))
off = cfg_row(v4, "kron13_2bit_gptq_sink", 8)

fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
seen = set()
OFFSET = {"Kronecker core, 2b1b, INT4 f.": (-10, 6, "right"), "Kronecker + e2e KL, 2-bit, INT4 f.": (8, -14, "left"),
          "Kronecker + e2e KL, 2-bit, INT8 f.": (8, 10, "left")}
for ser, lab, x, y, ci, col, mk in pts:
    ax.errorbar(x, y, yerr=[[y - ci[0]], [ci[1] - y]], fmt=mk, ms=8, color=col, ecolor=col, elinewidth=1.5, capsize=3,
                mec="#fcfcfb", mew=1.5, label=None if ser in seen else ser, zorder=3)
    seen.add(ser)
    dx, dy, ha = OFFSET.get(lab, (8, 4, "left"))
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK, ha=ha)
ax.plot(off["bpw"], 2.31, marker="^", ms=8, color=BLUE, mec="#fcfcfb", mew=1.5, clip_on=False, zorder=3)
ax.annotate(f"Kronecker core, 2-bit + sink rule, INT8 f.: +{off['dppl']:.2f} [{off['ci95'][0]:.2f}, {off['ci95'][1]:.2f}] (off scale)",
            (off["bpw"], 2.31), xytext=(10, -4), textcoords="offset points", fontsize=8, color=INK)
ax.set_xlim(2.0, 4.5); ax.set_ylim(0, 2.35)
ax.set_xlabel("bits per routed-expert FFN weight (codes + factors + scales + gains)", color=INK)
ax.set_ylabel("Δ perplexity vs FP16 (WikiText-2 valid slice, 24×256 tok)", color=INK)
ax.set_title("granite-3.1-3b-a800m-base, all 32 MoE FFN layers compressed — single runs, 95% bootstrap CIs",
             fontsize=10, color=INK)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)
ax.tick_params(colors=MUTED)
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
out = os.path.join(ROOT, "figures", "frontier_round8.png")
fig.tight_layout(); fig.savefig(out); print("wrote", out)
