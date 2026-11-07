# Next steps

Compression work is paused after this write-up. This is the roadmap for when it resumes.

## 1. Noah & Claude (chat) — as agreed

1. **Replicate the headline.** Run low-rank 2b1b with INT8 factors at least 3 times, with a larger eval (full
   WikiText-2 test plus a C4 slice) and zero-shot tasks (ARC, HellaSwag, BoolQ via lm-eval-harness). Report the
   run-to-run spread.
2. **Fix INT4 factors.** Use per-row or per-group scales for the low-rank U and V, or keep INT8, since at rank 11
   the factor cost is small (0.230 bpw). Re-measure the ~3.19 bpw point.
3. **Combine the two winners:** low-rank core + 2-bit residual + end-to-end KL distillation. This is the most likely
   path to a real Milestone B: ≤ 2.5 bpw at INT4 quality on **KL and top-1**, not just on perplexity.
4. **Put rank where the spectrum is.** Allocate rank per matrix and per layer by SVD spike energy (gate and down have
   structure, up much less). Allocate residual bits per layer and per matrix by calibration sensitivity.
5. **Vector or trellis codes** for the rotated residual (E8 / QTIP-style), to close the gap between the scalar
   quantizer and the rate-distortion bound (2-bit: 88.25% Lloyd-Max vs 93.75% R-D in weight R²).
6. **Temporal error.** The compression error is concentrated at low temporal frequency (layer 12). Try a small
   low-rank activation-space correction fitted to the slow error components.
7. **Generalize** to the dense `granite-3.1-2b-base`, then Noah's usual 8B. Pre-capture teacher activations and free
   the original weights to fit in 32 GB.
8. **Published baselines on the same model.** Run QuIP#, AQLM and QTIP code where available, at matched bpw.
9. **RMT spectral shrinkage** (Gavish–Donoho style, treating both spectral tails as signal) as a training-free
   denoising probe of the AdamW-entropy hypothesis, with a random-basis placebo.
10. **Longer term (Noah's thesis):** once compression is solid, test whether training *in* the low-rank + residual
    parameterization speeds up from-scratch training at matched compute on a small GPT.

## 2. Claude Code (agent) — additional ideas and issues seen in the code and runs

### Ideas (what, why, what would falsify it)

1. **Drop the core where there is no spectrum.** The up matrix has little low-rank structure (top-5% SVD energy
   0.0876 at layer 5, below the Gaussian 0.1139), so its rank-11 factors buy little. Spend those bits on up's
   residual instead, or reallocate the rank to gate and down.
   *Falsified if* low-rank on gate/down only, with the saved bits moved to residual or rank, is not at least as good
   as uniform rank 11 at equal bpw.
2. **Protect the sink direction explicitly.** Layer 3's sink output is reproduced at 99.5% open-loop R² by the
   low-rank core and at 16.2% by the Kronecker core, which suggests it lies along a few singular directions. Keep the
   top 1–2 singular pairs of the sink-layer down matrices in FP16, outside the quantized core.
   *Falsified if* adding them to the Kronecker 2-bit config does not raise layer-3 open-loop R² above ~50%.
3. **Activation-aware low-rank initialization.** Fit the low-rank SVD to W·diag(activation RMS) or to the
   Hessian-weighted error, ASVD/SVD-LLM style. Whitening hurt the Kronecker core on the 1B, but there the Kronecker
   rearrangement may have broken the benefit.
   *Falsified if* low-rank 2b1b with whitened initialization is not better (paired) at equal bpw.
4. **Mixed residual widths at about 2.7 bpw.** Instead of jumping from 2-bit (2.35 bpw) to 2b1b (3.31 bpw), give
   2b1b only to the matrices and layers with the largest calibration sensitivity. This tests whether the
   low-rank pipeline has a usable point between the two.
5. **Guard end-to-end KL against domain fitting.** Round 7 improved perplexity more than fidelity: KL to FP16 stayed
   2.8× RTN INT4's. Monitor KL on an out-of-domain slice during training and stop on it, or add a small weight-space
   anchor.
   *Falsified if* out-of-domain KL tracks in-domain KL anyway.
6. **Hessian shrinkage for starved experts.** Several experts see 0–107 calibration tokens per layer (12,288 tokens
   in total), and GPTQ then overfits their Hessians. Shrink each expert Hessian toward the layer's pooled Hessian, or
   use nearest rounding below a token threshold. Separately, collect more calibration tokens.

### Bugs, fragility and evaluation weaknesses noticed

- **Silent fallbacks.** `extract_granite_moe_weights` (scripts/rpm_ihog_15bpw_hadamard_distill.py) falls back to
  `top_k = 8` when it cannot find the attribute. That's correct for both Granite MoEs used here, but wrong for any
  model with a different top-k. The corpus loader silently falls back to TinyShakespeare if the WikiText download
  fails.
- **Clamped output gain hides failures.** Block initialization clamps the fitted output gain to [0.05, 20]. When a
  block is badly wrong (the rank-0 configuration) the gain sits at 0.05 instead of raising an error.
- **Factor quantizer** uses one absmax scale per factor term. That works for 32×32 Kronecker blocks and destroys
  low-rank vectors (see RESULTS.md).
- **In-sample damping selection.** The GPTQ INT3 control picks its damping on the same calibration text its
  Hessians come from, which favours the smallest damping. The pipeline's own damping was chosen on a calibration
  split. Use a split for every method.
- **Evaluation design.** 6,144 eval tokens adjacent to the calibration text in the same articles, a single domain,
  and no downstream tasks. Store per-sequence NLLs for every row. Rounds 5–7 did not, so their paired tests cannot
  be recomputed. Report KL and top-1 next to ΔPPL as primary metrics.
- **Weak INT4 bar.** RTN INT4 g64 (symmetric, no zero point) is a weak INT4 reference. GPTQ INT4 asym g128 would be
  the fairer "INT4 quality" target for Milestone B.
- **Nondeterminism.** Training is not bit-reproducible (sd ≈ 0.09 ΔPPL across three runs of one configuration). The
  2-bit per-layer stage is far less stable (+3.132 vs +1.688 for the same configuration). Either use deterministic
  kernels, or report the mean of several runs.
- **Efficiency.** `capture_real_moe_io` runs a full forward pass of the model for every layer and every capture.
  One pass with hooks on all layers would cut the per-layer overhead. The GPTQ column loop is Python-bound: the GPU
  sat at about 47 W during the INT3 controls.
- **Crashed runs.** Round 8 v2 and v3 stopped part-way (a memory leak in the driver and an ill-conditioned control
  configuration, both since addressed). Their reported rows were saved incrementally and are complete, but the
  in-run summary blocks (bpw matching, paired tests) were never written. They are recomputed by
  `scripts/analysis/paired_tests.py`.
