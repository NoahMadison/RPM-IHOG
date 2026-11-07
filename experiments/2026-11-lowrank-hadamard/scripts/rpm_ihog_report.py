#!/usr/bin/env python3
r"""
rpm_ihog_report.py — figures + multi-page PDF + REPORT.md from a round-5 results.json

  python rpm_ihog_report.py ./rpm_ihog_round5_granite-3.1-3b-a800m-base/results.json

Writes into <results dir>/report/:
  fig01_frontier.png          quality vs size: dPPL vs bpw with 95% CIs, RTN INT-k reference frontier
  fig02_bpw_breakdown.png     where the bits go (residual codes / Kronecker factors / scales)
  fig03_r2_depth.png          layer R^2 vs depth: open, closed-loop, one-shot (clean and drifted)
  fig04_weight_r2.png         weight-space R^2 per matrix vs depth, with scalar quantizer ceilings
  fig05_swap_sensitivity.png  output R^2 when only one matrix is compressed (which matrix hurts)
  fig06_cascade.png           closed-loop penalty (open - closed R^2) vs depth, with sink layers
  fig07_token_error.png       per-token relative error quantiles vs depth (+ sink token)
  fig08_training.png          distillation curves, every layer, coloured by depth
  RPM_IHOG_report.pdf         title + summary table + every figure with a caption (+ spectral pages)
  REPORT.md                   the same table and figure index in markdown
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# Okabe-Ito (colour-blind safe)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MATRIX_COLORS = {"gate": "#D55E00", "up": "#0072B2", "down": "#009E73"}
FB_MARKERS = {16: "o", 8: "s", 4: "D"}
SINK_SHADE = "#E8E8E8"
# Lloyd-Max scalar quantizer on a Gaussian source: 1 - D/sigma^2 (weight-space ceiling for the residual alone)
CEILING = {"1bit": 63.7, "tern": 81.0, "2bit": 88.25, "2b1b": 95.7}


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 180, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.fontsize": 8.5, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 5, "figure.constrained_layout.use": True,
    })


def cfg_colors(res):
    return {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(res["configs"])}


def layer_arrays(cfg, key):
    ls = sorted(int(l) for l in cfg["layers"])
    return ls, [cfg["layers"][str(l)].get(key, float("nan")) for l in ls]


def sink_layers(res, thresh=0.5):
    out = set()
    for cfg in res["configs"].values():
        for l, m in cfg["layers"].items():
            if m.get("y_pos0", 0) > thresh:
                out.add(int(l))
    return sorted(out)


def shade_sinks(ax, sinks):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    for s in sinks:
        ax.axvspan(s - 0.5, s + 0.5, color=SINK_SHADE, zorder=0, lw=0)


def grid_axes(n, w=5.2, h=3.6, sharey=True):
    cols = min(n, 2) if n <= 4 else 3
    rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(w * cols, h * rows), sharey=sharey, squeeze=False)
    flat = axs.flatten()
    for a in flat[n:]:
        a.set_visible(False)
    return fig, flat[:n]


def fb_label(fb):
    return "fp16 factors" if fb == 16 else f"INT{fb} factors"


# ------------------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------------------
def fig_frontier(res, cc):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    rtn = sorted(res["rtn"], key=lambda r: r["bpw"])
    if rtn:
        x = [r["bpw"] for r in rtn]
        y = [r["dppl"] for r in rtn]
        lo = [r["dppl"] - r["ci95"][0] for r in rtn]
        hi = [r["ci95"][1] - r["dppl"] for r in rtn]
        ax.errorbar(x, y, yerr=[lo, hi], color="black", marker="s", ms=7, capsize=3, lw=1.2, ls="--",
                    label="RTN INT-k, group 64 (reference)")
        for r in rtn:
            ax.annotate(r["name"], (r["bpw"], r["dppl"]), textcoords="offset points", xytext=(6, 4),
                        fontsize=8)
    for c, cfg in res["configs"].items():
        rows = sorted(cfg["rows"], key=lambda r: r["bpw"])
        x = [r["bpw"] for r in rows]
        y = [r["dppl"] for r in rows]
        ax.plot(x, y, color=cc[c], lw=1.4, alpha=0.8)
        for r in rows:
            ax.errorbar(r["bpw"], r["dppl"],
                        yerr=[[r["dppl"] - r["ci95"][0]], [r["ci95"][1] - r["dppl"]]],
                        color=cc[c], marker=FB_MARKERS.get(r["fb"], "o"), ms=7, capsize=3, mec="white", mew=0.8)
    ax.set_yscale("symlog", linthresh=0.05, linscale=0.6)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("bits per FFN weight (all overheads included)")
    ax.set_ylabel("Δ perplexity vs FP16  (symlog: linear within ±0.05, lower is better)")
    ax.set_title(f"Quality vs size, all {len(res['layers'])} FFN layers compressed — {res['model'].split('/')[-1]} "
                 f"(FP16 PPL {res['fp16_ppl']})")
    handles = [Line2D([], [], color=cc[c], lw=2, label=c) for c in res["configs"]]
    handles += [Line2D([], [], color="gray", marker=m, ls="", ms=7, label=fb_label(fb)) for fb, m in FB_MARKERS.items()]
    if rtn:
        handles.append(Line2D([], [], color="black", marker="s", ls="--", label="RTN INT-k g64"))
    ax.legend(handles=handles, loc="upper right", ncol=2)
    ax.text(0.01, 0.01, "error bars: 95% bootstrap CI over held-out sequences", transform=ax.transAxes,
            fontsize=8, color="gray")
    return fig


def fig_breakdown(res, cc):
    cats = ["residual codes", "Kronecker factors", "scales & gains"]
    cols = ["#0072B2", "#E69F00", "#999999"]
    labels, vals = [], {k: [] for k in cats}
    for c, cfg in res["configs"].items():
        for r in sorted(cfg["rows"], key=lambda r: -r["fb"]):
            labels.append(f"{c}\n{fb_label(r['fb']).replace(' factors', '')}")
            for k in cats:
                vals[k].append(r["breakdown"][k])
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(labels) + 2), 4.8))
    bottom = [0.0] * len(labels)
    for k, col in zip(cats, cols):
        ax.bar(range(len(labels)), vals[k], bottom=bottom, color=col, label=k, width=0.72)
        bottom = [b + v for b, v in zip(bottom, vals[k])]
    for i, b in enumerate(bottom):
        ax.text(i, b + 0.03, f"{b:.2f}", ha="center", fontsize=8)
    for r in res["rtn"]:
        ax.axhline(r["bpw"], color="black", ls=":", lw=0.8)
        ax.text(len(labels) - 0.5, r["bpw"], f" {r['name']}", va="center", fontsize=8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("bits per weight")
    ax.set_title("Where the bits go")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def fig_r2_depth(res, cc, sinks):
    fig, axs = grid_axes(len(res["configs"]))
    lo_clip = -60
    for ax, (c, cfg) in zip(axs, res["configs"].items()):
        shade_sinks(ax, sinks)
        series = [("open_r2_nopos0", "open (clean inputs, token 0 excluded)", "-", cc[c]),
                  ("closed_r2", "closed loop (compressed upstream)", "-", "#555555"),
                  ("oneshot_open", "one-shot, clean inputs", "--", cc[c]),
                  ("oneshot_closed", "one-shot, drifted inputs", ":", "#555555")]
        for key, lab, ls, col in series:
            l, v = layer_arrays(cfg, key)
            ax.plot(l, [max(x, lo_clip) for x in v], ls=ls, color=col, marker="o" if ls == "-" else None,
                    ms=3, label=lab)
            for li, x in zip(l, v):
                if x < lo_clip:
                    ax.plot(li, lo_clip, marker="v", color=col, ms=6)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(lo_clip - 5, 102)
        ax.set_title(c)
        ax.set_xlabel("layer")
        ax.set_ylabel("output R² (%)")
    axs[0].legend(loc="lower left")
    fig.suptitle(f"Layer-wise output fidelity vs depth (▼ = clipped below {lo_clip}%; grey bands = sink-writing layers)",
                 fontweight="bold")
    return fig


def fig_weight_r2(res, cc, sinks):
    fig, axs = grid_axes(len(res["configs"]))
    for ax, (c, cfg) in zip(axs, res["configs"].items()):
        shade_sinks(ax, sinks)
        seen = set()
        for mname, col in MATRIX_COLORS.items():
            l, v = layer_arrays(cfg, f"wR2_{mname}")
            ax.plot(l, v, color=col, marker="o", ms=3, label=f"{mname} ({cfg['residuals'][mname]})")
            r = cfg["residuals"][mname]
            if r in CEILING and r not in seen:
                ax.axhline(CEILING[r], color="black", ls="--", lw=0.8)
                ax.text(l[-1] + 0.3, CEILING[r], f"{r} scalar ceiling", fontsize=7.5, va="center")
                seen.add(r)
        ax.set_ylim(min(40, ax.get_ylim()[0]), 100)
        ax.set_title(c)
        ax.set_xlabel("layer")
        ax.set_ylabel("weight-space R² (%)")
        ax.legend(loc="lower left")
    fig.suptitle("Weight-space fidelity per matrix. Dashed: Lloyd-Max limit of the residual quantizer on a Gaussian "
                 "source; the Kronecker core is what can lift a matrix above it", fontweight="bold")
    return fig


def fig_swap(res, cc, sinks):
    fig, axs = grid_axes(len(res["configs"]))
    for ax, (c, cfg) in zip(axs, res["configs"].items()):
        shade_sinks(ax, sinks)
        for mname, col in MATRIX_COLORS.items():
            l, v = layer_arrays(cfg, f"swap_{mname}")
            ax.plot(l, v, color=col, marker="o", ms=3, label=f"only {mname} compressed")
        l, v = layer_arrays(cfg, "open_r2_nopos0")
        ax.plot(l, v, color="black", lw=1.2, ls="--", label="all three compressed")
        ax.set_ylim(min(0, ax.get_ylim()[0]), 102)
        ax.set_title(c)
        ax.set_xlabel("layer")
        ax.set_ylabel("output R² (%), token 0 excluded")
    axs[0].legend(loc="lower left")
    fig.suptitle("Which matrix hurts? Output R² with exactly one matrix compressed, the other two exact",
                 fontweight="bold")
    return fig


def fig_cascade(res, cc, sinks):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    shade_sinks(ax, sinks)
    shade_sinks(ax2, sinks)
    for c, cfg in res["configs"].items():
        l, op = layer_arrays(cfg, "open_r2")
        _, cl = layer_arrays(cfg, "closed_r2")
        ax.plot(l, [max(o - k, -5) for o, k in zip(op, cl)], color=cc[c], marker="o", ms=3, label=c)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("open − closed R² (points)")
    ax.set_title("Cascade penalty: fidelity lost because upstream layers are compressed")
    ax.legend()
    any_cfg = next(iter(res["configs"].values()))
    l, y0 = layer_arrays(any_cfg, "y_pos0")
    ax2.bar(l, [100 * v for v in y0], color="#999999")
    ax2.set_ylabel("% output energy\non token 0")
    ax2.set_xlabel("layer")
    ax2.set_ylim(0, 105)
    ax2.set_title("Attention-sink signature (teacher)")
    return fig


def fig_token_error(res, cc, sinks):
    fig, axs = grid_axes(len(res["configs"]))
    for ax, (c, cfg) in zip(axs, res["configs"].items()):
        shade_sinks(ax, sinks)
        for key, ls, lab in (("relerr_p50", "-", "median token"), ("relerr_p90", "--", "90th pct"),
                             ("relerr_p99", ":", "99th pct")):
            l, v = layer_arrays(cfg, key)
            ax.plot(l, v, color=cc[c], ls=ls, label=lab)
        l, v = layer_arrays(cfg, "relerr_sink")
        ax.plot(l, v, ls="", marker="x", color="black", label="token 0 (sink)")
        ax.set_yscale("log")
        ax.set_title(c)
        ax.set_xlabel("layer")
        ax.set_ylabel("|ŷ − y| / |y| per token")
    axs[0].legend(loc="upper left")
    fig.suptitle("Per-token relative error of the FFN output (clean held-out inputs)", fontweight="bold")
    return fig


def fig_training(res, cc):
    fig, axs = grid_axes(len(res["configs"]))
    cmap = plt.get_cmap("viridis")
    nL = max(1, res["n_layers"] - 1)
    for ax, (c, cfg) in zip(axs, res["configs"].items()):
        for l, m in sorted(cfg["layers"].items(), key=lambda kv: int(kv[0])):
            h = m.get("hist") or []
            if h:
                ax.plot([p[0] for p in h], [p[1] for p in h], color=cmap(int(l) / nL), lw=0.9, alpha=0.85)
        ax.set_yscale("log")
        ax.set_title(c)
        ax.set_xlabel("distillation step")
        ax.set_ylabel("relative MSE (training batch)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, res["n_layers"] - 1))
    fig.colorbar(sm, ax=list(axs), label="layer", shrink=0.8)
    fig.suptitle("Distillation curves (vertical kinks at 30% / 60% = residual repacks)", fontweight="bold")
    return fig


# ------------------------------------------------------------------------------
# Tables / pages
# ------------------------------------------------------------------------------
def table_rows(res):
    rows = [["FP16", "16.00", f"{res['fp16_ppl']:.3f}", "—", "—", "—"]]
    for r in sorted(res["rtn"], key=lambda r: -r["bpw"]):
        rows.append([r["name"] + " g64", f"{r['bpw']:.2f}", f"{r['ppl']:.3f}", f"{r['dppl']:+.3f}",
                     f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]", "—"])
    for c, cfg in res["configs"].items():
        _, op = layer_arrays(cfg, "open_r2_nopos0")
        mean_open = sum(op) / len(op) if op else float("nan")
        for r in sorted(cfg["rows"], key=lambda r: -r["fb"]):
            rows.append([f"{c} ({fb_label(r['fb']).replace(' factors', '')} factors)", f"{r['bpw']:.2f}",
                         f"{r['ppl']:.3f}", f"{r['dppl']:+.3f}", f"[{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]",
                         f"{mean_open:.1f}"])
    return rows


HEAD = ["representation", "bpw", "PPL", "ΔPPL", "95% CI", "mean layer R² (%)"]


def title_page(res):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.06, 0.92, "RPM-IHOG compression report", fontsize=22, fontweight="bold")
    a = res["args"]
    meta = (f"Model: {res['model']}  ({res['kind']}, {res['n_layers']} layers, {len(res['layers'])} compressed)\n"
            f"Baseline FP16 perplexity: {res['fp16_ppl']}   |   Generated {res.get('created', '')}\n"
            f"Calibration {a.get('calib_seqs')}×{a.get('seq_len')} tokens, held-out {a.get('eval_seqs')}×"
            f"{a.get('seq_len')} tokens (WikiText-2), {a.get('steps')} distillation steps/layer, closed-loop.")
    fig.text(0.06, 0.82, meta, fontsize=10.5, va="top", linespacing=1.6)
    ax = fig.add_axes([0.06, 0.08, 0.88, 0.6])
    ax.axis("off")
    rows = table_rows(res)
    t = ax.table(cellText=rows, colLabels=HEAD, loc="upper center", cellLoc="center", colLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.35)
    for (r, cidx), cell in t.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#F2F2F2")
    return fig


def caption_page(fig, caption):
    fig.text(0.01, -0.02, caption, fontsize=8.5, color="#333333", wrap=True, va="top")
    return fig


CAPTIONS = {
    "fig01_frontier": "Δ perplexity (log scale) against storage for every configuration, all FFN layers compressed. "
                      "Black: plain round-to-nearest INT-k with group-64 scales. A point below and left of the RTN "
                      "curve beats it at equal or smaller size. Marker shape = precision of the Kronecker factors.",
    "fig02_bpw_breakdown": "Bits per weight split into residual codes, Kronecker factors and per-row/column scales and "
                           "gains. Dotted lines mark the RTN INT-k footprints.",
    "fig03_r2_depth": "Output R² of each compressed FFN block on held-out tokens. 'Open' = clean inputs, isolating "
                      "the layer's own error; 'closed' = inputs from the already-compressed model. One-shot = "
                      "initialisation only, before distillation. Grey bands: layers that write the attention sink.",
    "fig04_weight_r2": "1 − |W − Ŵ|² / |W|² per matrix. Dashed lines: best possible scalar quantizer of that residual "
                       "type on Gaussian data, i.e. what the bits alone can do.",
    "fig05_swap_sensitivity": "Output R² when exactly one of gate/up/down is replaced by its compressed version. "
                              "A low curve means that matrix dominates the block's error.",
    "fig06_cascade": "Top: R² lost purely because upstream layers are compressed. Bottom: share of the teacher's "
                     "output energy on the first token of each sequence (attention-sink signature).",
    "fig07_token_error": "Distribution over tokens of the relative output error |ŷ−y|/|y|; crosses = the sink token.",
    "fig08_training": "Relative MSE on training batches during distillation, one line per layer.",
}


def make_report(results_path):
    style()
    with open(results_path) as f:
        res = json.load(f)
    out = os.path.join(os.path.dirname(os.path.abspath(results_path)), "report")
    os.makedirs(out, exist_ok=True)
    cc = cfg_colors(res)
    sinks = sink_layers(res)
    figs = []
    if res["configs"] or res["rtn"]:
        figs.append(("fig01_frontier", fig_frontier(res, cc)))
    if res["configs"]:
        figs += [("fig02_bpw_breakdown", fig_breakdown(res, cc)), ("fig03_r2_depth", fig_r2_depth(res, cc, sinks)),
                 ("fig04_weight_r2", fig_weight_r2(res, cc, sinks)), ("fig05_swap_sensitivity", fig_swap(res, cc, sinks)),
                 ("fig06_cascade", fig_cascade(res, cc, sinks)), ("fig07_token_error", fig_token_error(res, cc, sinks)),
                 ("fig08_training", fig_training(res, cc))]
    pdf_path = os.path.join(out, "RPM_IHOG_report.pdf")
    with PdfPages(pdf_path) as pdf:
        tp = title_page(res)
        pdf.savefig(tp)
        plt.close(tp)
        for name, fig in figs:
            fig.savefig(os.path.join(out, f"{name}.png"), bbox_inches="tight")
            caption_page(fig, CAPTIONS[name])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        base = os.path.dirname(os.path.abspath(results_path))
        for l in sorted(res.get("spectral", {}), key=int):
            sdir = os.path.join(base, f"spectral_L{l}")
            for png in sorted(p for p in os.listdir(sdir) if p.endswith(".png")) if os.path.isdir(sdir) else []:
                img = plt.imread(os.path.join(sdir, png))
                fig = plt.figure(figsize=(11, 11 * img.shape[0] / img.shape[1] + 0.6))
                ax = fig.add_axes([0, 0.06, 1, 0.9])
                ax.imshow(img)
                ax.axis("off")
                fig.text(0.01, 0.01, f"Spectral analysis, layer {l}: {png[:-4].replace('_', ' ')} "
                                     "(controls: row/column-permuted teacher and variance-matched Gaussian)",
                         fontsize=9, color="#333333")
                pdf.savefig(fig)
                plt.close(fig)
        info = pdf.infodict()
        info["Title"] = f"RPM-IHOG report — {res['model']}"

    md = [f"# RPM-IHOG report — `{res['model']}`", "",
          f"{res['kind']}, {res['n_layers']} layers ({len(res['layers'])} compressed), FP16 PPL **{res['fp16_ppl']}**. "
          f"Sink-writing layers: {sinks or 'none detected'}.", "",
          "| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)]
    md += ["| " + " | ".join(r) + " |" for r in table_rows(res)]
    md += ["", "## Figures", ""]
    md += [f"- **{n}** — {CAPTIONS[n]}  \n  ![](report/{n}.png)" for n, _ in figs]
    for l in sorted(res.get("spectral", {}), key=int):
        md.append(f"- spectral analysis layer {l}: `spectral_L{l}/`")
    with open(os.path.join(os.path.dirname(out), "REPORT.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[+] report: {pdf_path}  ({len(figs)} figures, PNGs + REPORT.md alongside)")
    return pdf_path


if __name__ == "__main__":
    make_report(sys.argv[1])
