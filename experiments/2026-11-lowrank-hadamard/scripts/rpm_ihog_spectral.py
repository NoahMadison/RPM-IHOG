#!/usr/bin/env python3
r"""
rpm_ihog_spectral.py — what does the compression actually do, spectrally?
=========================================================================
Loads one layer's teacher FFN weights plus the compressed checkpoints that round 4 saved for
that layer (ckpt_<config>_L<layer>.pt), and compares "before" (teacher) with "after"
(reconstruction) and the error, in several domains. Every weight-space plot carries two controls:

  permuted : teacher with rows and columns randomly shuffled. A network is invariant to this, so
             any structure that disappears under permutation is an artifact of index order.
  gaussian : i.i.d. Gaussian with the teacher's variance. Rotation-invariant: it compacts equally
             badly in EVERY orthonormal basis, so "beats gaussian" = real structure in that basis.

Figures (PNG, in --out_dir):
  fig1_fft2d.png        2D FFT log-power: teacher / permuted / gaussian / recon / error
  fig2_radial_psd.png   radially averaged 2D power spectrum + spectral flatness (1.0 = white)
  fig3_compaction.png   energy captured by the top-p% coefficients in raw, FFT, Haar, WHT, SVD
                        domains (the key plot: which basis, if any, compacts the weights?)
  fig4_haar_bands.png   2D Haar wavelet energy per level (approximation vs detail bands)
  fig5_svd_kron.png     ordinary singular values and Kronecker (Van Loan) singular values
  fig6_wht_hist.png     residual distribution before vs after the Hadamard rotation (+ kurtosis)
  fig7_temporal_psd.png FFT along SEQUENCE POSITION of the FFN output: signal vs error, per config
                        (a physically meaningful axis: is the compression error white in time?)
  summary.json          the numbers behind the plots

Run after round 4 (it uses that run's checkpoints), e.g.
  python rpm_ihog_spectral.py --layer 12
"""

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import rpm_ihog_round3 as R3
import rpm_ihog_round4 as R4

COLORS = {"teacher": "#1f4e79", "permuted": "#8c8c8c", "gaussian": "#c9a227"}
CFG_COLORS = ["#c0392b", "#27ae60", "#8e44ad", "#d35400"]


# ------------------------------------------------------------------------------
# Transforms (all orthonormal so energies are comparable)
# ------------------------------------------------------------------------------
def haar2d(x, levels):
    """Returns list of (level, band, coeffs) plus the final approximation."""
    out, a = [], x
    for lv in range(1, levels + 1):
        if a.shape[0] < 2 or a.shape[1] < 2 or a.shape[0] % 2 or a.shape[1] % 2:
            break
        p, q, r, s = a[0::2, 0::2], a[0::2, 1::2], a[1::2, 0::2], a[1::2, 1::2]
        out += [(lv, "LH", (p - q + r - s) / 2), (lv, "HL", (p + q - r - s) / 2), (lv, "HH", (p - q - r + s) / 2)]
        a = (p + q + r + s) / 2
    out.append((len(out) // 3, "LL", a))
    return out


def domains(M):
    """Coefficient energies of M in several orthonormal bases (flattened)."""
    dev = M.device
    Ho, Hi = R4.block_hadamard(M.shape[0], dev), R4.block_hadamard(M.shape[1], dev)
    return {
        "raw": M.pow(2).flatten(),
        "fft2": torch.fft.fft2(M, norm="ortho").abs().pow(2).flatten(),
        "haar": torch.cat([c.pow(2).flatten() for _, _, c in haar2d(M, 5)]),
        "wht": (Ho @ M @ Hi).pow(2).flatten(),
        "svd": torch.linalg.svdvals(M).pow(2),
    }


def compaction(e, ps):
    e = torch.sort(e, descending=True).values
    c = torch.cumsum(e, 0) / e.sum()
    n = e.numel()
    return [float(c[max(0, min(n - 1, int(math.ceil(p * n)) - 1))]) for p in ps]


def radial_psd(M, nbins=48):
    P = torch.fft.fftshift(torch.fft.fft2(M, norm="ortho").abs().pow(2))
    h, w = P.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=M.device), torch.linspace(-1, 1, w, device=M.device),
                            indexing="ij")
    r = (yy ** 2 + xx ** 2).sqrt() / math.sqrt(2)
    b = (r * (nbins - 1)).round().long().flatten()
    acc = torch.zeros(nbins, device=M.device).index_add_(0, b, P.flatten())
    cnt = torch.zeros(nbins, device=M.device).index_add_(0, b, torch.ones_like(P.flatten()))
    flat = float(P.clamp_min(1e-30).log().mean().exp() / P.mean())
    return (acc / cnt.clamp_min(1)).cpu(), flat


def van_loan(M, p1=32, q1=32):
    p2, q2 = M.shape[0] // p1, M.shape[1] // q1
    R = M.reshape(p1, p2, q1, q2).permute(0, 2, 1, 3).reshape(p1 * q1, p2 * q2)
    return torch.linalg.svdvals(R).pow(2)


def kurtosis(x):
    x = x.flatten().float()
    x = (x - x.mean()) / x.std()
    return float(x.pow(4).mean())


# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------
@torch.no_grad()
def analyze(model, layer, ckpt_dir, configs, out_dir, evalt, dev, expert=None, seed=0, cfg_table=None):
    cfg_table = cfg_table or R4.CONFIGS
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device="cpu").manual_seed(seed)
    ffn = R4.get_ffn(model, layer)
    rw, w_in, w_out, tk = R4.extract(ffn, dev)
    E, two_f, d = w_in.shape
    f = two_f // 2

    # teacher activations on eval text (all-original model)
    X, Y, _ = R3.capture_real_moe_io(model, ffn, evalt, dev)
    S, T = evalt.shape[0], X.shape[0] // evalt.shape[0]

    if expert is None:
        _, ti = torch.topk(F.linear(X.float(), rw.float()), tk, dim=-1)
        expert = int(torch.bincount(ti.flatten(), minlength=E).argmax()) if E > 1 else 0
    teach = {"gate": w_in[expert, :f].float(), "up": w_in[expert, f:].float(), "down": w_out[expert].float()}

    blocks = {}
    for c in configs:
        path = os.path.join(ckpt_dir, f"ckpt_{c}_L{layer}.pt")
        if not os.path.exists(path):
            print(f"[!] missing {path} (run round 4 frontier for {c} first); skipping")
            continue
        spec = cfg_table[c]
        blk = spec.get("block_cls", R3.Block3)(E, d, f, tk, spec["rank"], spec["res"]).to(dev)
        blk.load_state_dict(torch.load(path, map_location=dev, weights_only=True), strict=True)
        blk.eval()
        blocks[c] = blk
    if not blocks:
        raise SystemExit("no checkpoints found; run rpm_ihog_round4.py --exps frontier first")

    recon = {c: {"gate": b.gate.synthesize()[expert], "up": b.up.synthesize()[expert],
                 "down": b.down.synthesize()[expert]} for c, b in blocks.items()}

    def controls(M):
        pr = torch.randperm(M.shape[0], generator=g).to(dev)
        pc = torch.randperm(M.shape[1], generator=g).to(dev)
        return M[pr][:, pc], torch.randn(M.shape, generator=g).to(dev) * M.std()

    summary = {"layer": layer, "expert": expert, "configs": list(blocks), "matrices": {}}
    ps = [10 ** (k / 20) for k in range(-60, 1)]   # 0.1% .. 100%

    # ---- fig1: 2D FFT log-power (gate) ----
    M = teach["gate"]
    Mp, Mg = controls(M)
    c0 = next(iter(blocks))
    panels = [("teacher", M), ("permuted", Mp), ("gaussian", Mg),
              (f"recon ({c0})", recon[c0]["gate"]), (f"error ({c0})", recon[c0]["gate"] - M)]
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    for a, (name, A) in zip(ax, panels):
        P = torch.fft.fftshift(torch.fft.fft2(A, norm="ortho").abs().pow(2)).clamp_min(1e-30).log10().cpu()
        im = a.imshow(P, aspect="auto", cmap="magma", vmin=float(P.quantile(0.01)), vmax=float(P.quantile(0.999)))
        a.set_title(name)
        a.set_xticks([]), a.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.8, label="log10 power")
    fig.suptitle(f"Layer {layer}, expert {expert}, gate: 2D FFT (index axes are permutable!)")
    fig.savefig(os.path.join(out_dir, "fig1_fft2d.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- fig2/3/4/5 per matrix ----
    fig2, ax2 = plt.subplots(1, 3, figsize=(18, 4.6), layout="constrained")
    fig3, ax3 = plt.subplots(3, 5, figsize=(24, 13), sharey=True, layout="constrained")
    fig4, ax4 = plt.subplots(1, 3, figsize=(18, 4.6), layout="constrained")
    fig5, ax5 = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")
    for j, (mname, M) in enumerate(teach.items()):
        Mp, Mg = controls(M)
        srcs = {"teacher": M, "permuted": Mp, "gaussian": Mg}
        errs = {c: recon[c][mname] - M for c in blocks}
        ms = {"flatness": {}, "compaction_at_5pct": {}, "kron13_energy": {}, "svd_top5pct_energy": {}}

        for name, A in list(srcs.items()) + [(f"error {c}", e) for c, e in errs.items()]:
            psd, flat = radial_psd(A)
            ms["flatness"][name] = round(flat, 4)
            col = COLORS.get(name, CFG_COLORS[list(blocks).index(name[6:]) % 4] if name.startswith("error") else "k")
            ax2[j].semilogy(torch.linspace(0, 1, len(psd)), psd, label=f"{name} (flat {flat:.2f})", color=col,
                            ls="--" if name.startswith("error") else "-")
        ax2[j].set_title(f"{mname}: radial 2D power spectrum")
        ax2[j].set_xlabel("normalized spatial frequency")
        ax2[j].legend(fontsize=7)

        for name, A in srcs.items():
            dom = domains(A)
            for k, (dname, e) in enumerate(dom.items()):
                cv = compaction(e, ps) if dname != "svd" else compaction(e, ps)
                ax3[j, k].semilogx([p * 100 for p in ps], cv, label=name, color=COLORS[name])
                ax3[j, k].set_title(f"{mname} — {dname}")
                ms["compaction_at_5pct"].setdefault(dname, {})[name] = round(compaction(e, [0.05])[0], 4)
        for k in range(5):
            ax3[j, k].set_xlabel("top p% of coefficients")
            ax3[j, k].grid(alpha=0.3)
        ax3[j, 0].set_ylabel("fraction of energy")
        ax3[0, 0].legend()

        width = 0.27
        for i, (name, A) in enumerate(srcs.items()):
            bands = haar2d(A, 5)
            tot = sum(float(c.pow(2).sum()) for _, _, c in bands)
            labels = [f"L{lv}{b}" if b != "LL" else "LL" for lv, b, _ in bands]
            vals = [float(c.pow(2).sum()) / tot for _, _, c in bands]
            xs = [x + (i - 1) * width for x in range(len(vals))]
            ax4[j].bar(xs, vals, width, label=name, color=COLORS[name])
            ms.setdefault("haar_band_energy", {})[name] = dict(zip(labels, [round(v, 4) for v in vals]))
        ax4[j].set_xticks(range(len(labels)))
        ax4[j].set_xticklabels(labels, rotation=60, fontsize=7)
        ax4[j].set_title(f"{mname}: Haar band energy (white noise -> 1/4 per band at L1)")
        ax4[j].legend(fontsize=8)

        for name, A in srcs.items():
            sv = torch.linalg.svdvals(A).pow(2)
            sv = sv / sv.sum()
            ax5[0, j].loglog(range(1, len(sv) + 1), sv.cpu(), label=name, color=COLORS[name])
            ms["svd_top5pct_energy"][name] = round(compaction(sv, [0.05])[0], 4)
            vl = van_loan(A)
            vl = vl / vl.sum()
            ax5[1, j].loglog(range(1, len(vl) + 1), vl.cpu(), label=name, color=COLORS[name])
            ms["kron13_energy"][name] = round(float(vl[:13].sum()), 4)
        ax5[0, j].set_title(f"{mname}: singular value energy")
        ax5[1, j].set_title(f"{mname}: Kronecker (Van Loan) spectrum — rank-13 captures "
                            f"{ms['kron13_energy']['teacher']:.1%} vs gaussian {ms['kron13_energy']['gaussian']:.1%}",
                            fontsize=9)
        for a in ax5[:, j]:
            a.legend(fontsize=8)
            a.grid(alpha=0.3)
        summary["matrices"][mname] = ms

    for fig, name in ((fig2, "fig2_radial_psd"), (fig3, "fig3_compaction"), (fig4, "fig4_haar_bands"),
                      (fig5, "fig5_svd_kron")):
        fig.suptitle(f"Layer {layer}, expert {expert} — teacher vs permuted vs gaussian controls")
        fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ---- fig6: residual before/after Hadamard rotation (what the 1/2-bit grid actually sees) ----
    b = blocks[c0].gate
    scale = torch.exp(b.log_r.float()) * torch.exp(b.log_c.float())
    K = b.kron(torch.float32)
    res = (w_in[:, :f].float() / scale - (K if K is not None else 0.0))
    rot = R4.block_hadamard(res.shape[1], dev) @ res @ R4.block_hadamard(res.shape[2], dev)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for a, (name, A) in zip(ax, (("raw residual", res), ("Hadamard-rotated residual", rot))):
        z = (A.flatten() - A.mean()) / A.std()
        z = z[torch.randperm(z.numel(), generator=g)[:400_000].to(dev)].cpu()
        a.hist(z, bins=200, density=True, color="#1f4e79", alpha=0.75, range=(-6, 6))
        xx = torch.linspace(-6, 6, 400)
        a.plot(xx, torch.exp(-xx ** 2 / 2) / math.sqrt(2 * math.pi), "r--", label="N(0,1)")
        a.set_yscale("log")
        a.set_ylim(1e-6, 1)
        kt = kurtosis(A)
        a.set_title(f"{name}: kurtosis {kt:.2f} (Gaussian = 3)")
        a.legend()
        summary.setdefault("residual_kurtosis", {})[name] = round(kt, 3)
    fig.suptitle(f"Layer {layer} gate residual after the {c0} Kronecker core, all experts")
    fig.savefig(os.path.join(out_dir, "fig6_wht_hist.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- fig7: temporal (sequence-position) spectrum of FFN output: signal vs error ----
    Yt = Y.float().reshape(S, T, d)
    Pt = torch.fft.rfft(Yt - Yt.mean(1, keepdim=True), dim=1).abs().pow(2).mean((0, 2)).cpu()
    fq = torch.fft.rfftfreq(T)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    ax[0].semilogy(fq, Pt, color=COLORS["teacher"], label="teacher FFN output")
    for i, (c, blk) in enumerate(blocks.items()):
        Ys = torch.cat([blk.forward_moe(X[k : k + 2048].float()) for k in range(0, X.shape[0], 2048)]).reshape(S, T, d)
        Er = Ys - Yt
        Pe = torch.fft.rfft(Er - Er.mean(1, keepdim=True), dim=1).abs().pow(2).mean((0, 2)).cpu()
        ax[0].semilogy(fq, Pe, color=CFG_COLORS[i % 4], ls="--", label=f"error {c}")
        ax[1].plot(fq, 10 * torch.log10(Pt / Pe), color=CFG_COLORS[i % 4], label=c)
        summary.setdefault("temporal_snr_db", {})[c] = {
            "low_freq": round(float(10 * torch.log10(Pt[1:8].sum() / Pe[1:8].sum())), 2),
            "high_freq": round(float(10 * torch.log10(Pt[-32:].sum() / Pe[-32:].sum())), 2),
        }
    ax[0].set_title("power along sequence position (per channel, averaged)")
    ax[0].set_xlabel("cycles / token")
    ax[0].legend(fontsize=8)
    ax[1].set_title("SNR vs temporal frequency (flat = error is white in time)")
    ax[1].set_xlabel("cycles / token")
    ax[1].set_ylabel("dB")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f"Layer {layer}: FFN output, teacher vs compressed (closed-loop checkpoints, clean inputs)")
    fig.savefig(os.path.join(out_dir, "fig7_temporal_psd.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--expert", type=int, default=None, help="default: most-routed expert on eval text")
    ap.add_argument("--configs", nargs="+", default=list(R4.CONFIGS), choices=list(R4.CONFIGS))
    ap.add_argument("--ckpt_dir", default=None, help="default: round 4's out_dir for this model")
    ap.add_argument("--corpus_dir", default="./rpm_ihog_15bpw_results")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    tag = args.model_id.split("/")[-1]
    args.ckpt_dir = args.ckpt_dir or f"./rpm_ihog_round4_{tag}"
    args.out_dir = args.out_dir or f"./rpm_ihog_spectral_{tag}_L{args.layer}"

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    _, evalt, _ = R4.V1.load_disjoint_calib_and_eval_tokens(tok, 256, 48, 24, cache_dir=args.corpus_dir)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.float16).to(dev).eval()
    s = analyze(model, args.layer, args.ckpt_dir, args.configs, args.out_dir, evalt, dev, args.expert)
    print(json.dumps(s, indent=1)[:3000])
    print(f"[+] plots in {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
