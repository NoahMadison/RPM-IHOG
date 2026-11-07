# Results — full tables

Model `ibm-granite/granite-3.1-3b-a800m-base` (MoE, 32 layers, 40 experts, top-8), all 32 routed-expert FFN
layers compressed unless stated. ΔPPL is against the same run's FP16 model (FP16 PPL 6.928–6.935 across runs; GPU
nondeterminism). Eval: 24 sequences × 256 tokens of WikiText-2 *validation* text; calibration: the first 48 × 256
tokens of the same file (eval starts 4 sequences after calibration ends — disjoint, adjacent). 95% CIs are bootstrap
over sequences (4000 resamples). **bpw** = bits per routed-expert FFN weight, counting residual codes, core factors
(plus one fp16 scale per factor term when factors are quantized), row/column scales, residual gains and the output
gain. Routers, attention, embeddings and norms stay FP16 and are not counted.

Naming: `kron13` = Kronecker core of rank 13 (sum of 13 products A⊗B, A 32×32); `lr` = low-rank core (sum of
r outer products, r = 11); `2bit` = 2-bit Lloyd-Max residual; `2b1b` = 2-bit + 1-bit sign residual (3 bits);
`_gptq` = residual codes chosen by GPTQ/LDLQ in the Hadamard domain; `f16/INT8/INT4` = core-factor precision.
top-1 = agreement of the argmax token with FP16; KL = mean per-token KL(p_FP16 ‖ p_compressed); R²h = R² of the
final hidden state vs FP16. Source files are under `results/`.

## Round 8 v3 — the controlled comparison (single run; `results/round8_v3_results.json`)

| row | bpw | ΔPPL | 95% CI | median per-seq ΔNLL | top-1 % | KL/tok | R²h % |
|---|---|---|---|---|---|---|---|
| RTN INT4 g64 | 4.25 | +0.445 | [0.324, 0.571] | +0.0656 | 88.42 | 0.0912 | 91.43 |
| RTN INT3 g64 | 3.25 | +1.905 | [1.615, 2.218] | +0.2389 | 76.78 | 0.3466 | 68.37 |
| GPTQ INT3 asym g64, damp 0.01 (selected on calibration NLL) | 3.297 | +0.661 | [0.542, 0.789] | +0.0772 | 84.61 | 0.1461 | 85.15 |
| GPTQ INT3 asym g64, damp 0.1 | 3.297 | +0.559 | [0.447, 0.689] | +0.0690 | 86.37 | 0.1284 | 86.87 |
| GPTQ INT3 asym g64, damp 1.0 | 3.297 | +0.612 | [0.503, 0.737] | +0.0803 | 85.39 | 0.1420 | 86.17 |
| kron13_2b1b_gptq, f16 | 3.550 | +0.476 | [0.363, 0.584] | +0.0637 | 86.18 | 0.1363 | 88.04 |
| kron13_2b1b_gptq, INT8 | 3.314 | +0.478 | [0.365, 0.588] | +0.0611 | 86.19 | 0.1362 | 88.05 |
| kron13_2b1b_gptq, INT4 | 3.195 | +0.533 | [0.416, 0.654] | +0.0668 | 85.82 | 0.1473 | 86.65 |
| **lr_2b1b_gptq, f16** | 3.535 | **+0.098** | [0.014, 0.172] | +0.0049 | 87.45 | 0.0908 | 89.92 |
| **lr_2b1b_gptq, INT8** | 3.306 | **+0.095** | [0.014, 0.166] | +0.0037 | 87.32 | 0.0906 | 89.94 |
| lr_2b1b_gptq, INT4 | 3.191 | +68003 | [37956.543, 98452.391] | +9.4213 | 1.21 | 9.3121 | −379.41 |

GPTQ INT3 asym = static groups of 64, fp16 scale + 3-bit zero point (3 + 19/64 bpw), no rotation, no core, same
per-expert Hessians as the pipeline. Its damp was chosen on calibration NLL, which is in-sample for the Hessian and
picked the smallest damp; damp 0.1 is better on the eval set and is reported but not selected.
The low-rank INT4 row fails because the factor quantizer uses one absmax scale per factor vector (512 or 1536
entries); see NEXT_STEPS.md.

### Paired comparisons (same 24 sequences; `results/paired_tests.json`)

PPL(row) − PPL(ref), 10,000 paired resamples; covers eval-sampling noise only, not run-to-run training noise.

| row | ref | bpw | paired ΔPPL | 95% CI | P(row worse) |
|---|---|---|---|---|---|
| lr_2b1b_gptq INT8 | kron13_2b1b_gptq INT8 | 3.306 | −0.382 | [−0.473, −0.296] | 0.0 |
| lr_2b1b_gptq f16 | kron13_2b1b_gptq INT8 | 3.535 | −0.380 | [−0.462, −0.303] | 0.0 |
| GPTQ INT3 asym d0.01 | kron13_2b1b_gptq INT8 | 3.297 | +0.183 | [0.014, 0.369] | 0.982 |
| GPTQ INT3 asym d0.1 | kron13_2b1b_gptq INT8 | 3.297 | +0.081 | [−0.062, 0.242] | 0.861 |
| GPTQ INT3 asym d1.0 | kron13_2b1b_gptq INT8 | 3.297 | +0.135 | [−0.01, 0.295] | 0.964 |
| RTN INT4 | kron13_2b1b_gptq INT8 | 4.25 | −0.033 | [−0.201, 0.149] | 0.372 |
| RTN INT3 | kron13_2b1b_gptq INT8 | 3.25 | +1.427 | [1.114, 1.775] | 1.0 |
| kron13_2b1b_gptq INT4 | kron13_2b1b_gptq INT8 | 3.195 | +0.055 | [0.004, 0.103] | 0.982 |

## Round 8 v4 — 2-bit residual with the sink rule (single run; `results/round8_v4_results.json`)

Sink rule: in layers whose teacher FFN output puts > 50% of its energy on token 0 (measured on calibration text:
layers 3, 5, 12, 13, 29, 30 with shares 0.9, 0.894, 0.589, 0.5866, 0.787, 0.8251), the down matrix uses 2b1b
instead of 2-bit and its codes are frozen after initialization. Cost: +0.0645 bpw.

| row | bpw | ΔPPL | 95% CI | median per-seq ΔNLL | top-1 % | KL/tok |
|---|---|---|---|---|---|---|
| RTN INT4 g64 | 4.25 | +0.446 | [0.326, 0.57] | +0.0645 | 88.51 | 0.091 |
| RTN INT3 g64 | 3.25 | +1.900 | [1.613, 2.211] | +0.2381 | 76.47 | 0.3451 |
| **lr_2bit_gptq (sink rule), f16** | 2.582 | +0.602 | [0.41, 0.801] | +0.0783 | 81.09 | 0.2173 |
| **lr_2bit_gptq (sink rule), INT8** | 2.353 | **+0.618** | [0.425, 0.821] | +0.0847 | 80.95 | 0.2187 |
| lr_2bit_gptq (sink rule), INT4 | 2.239 | +70324 | [55034.336, 87391.945] | +9.2856 | 0.46 | 9.3041 |
| kron13_2bit_gptq_sink, f16 | 2.597 | +5.024 | [2.665, 11.034] | +0.3227 | 67.89 | 0.7213 |
| kron13_2bit_gptq_sink, INT8 | 2.361 | +5.108 | [2.678, 11.364] | +0.3215 | 67.88 | 0.7273 |
| kron13_2bit_gptq_sink, INT4 | 2.243 | +5.368 | [2.873, 11.854] | +0.344 | 67.81 | 0.7464 |

Paired (same sequences, same run): lr_2bit_gptq INT8 vs RTN INT4 +0.172 [−0.088, 0.425] (P(worse) 0.899);
vs RTN INT3 −1.282 [−1.622, −0.99]. kron13_2bit_gptq_sink INT8 vs RTN INT4 +4.662 [2.179, 10.936].

## Round 8 v2 — controls and a replicate (single run; `results/round8_v2_results.json`)

| row | bpw | ΔPPL | 95% CI | top-1 % | KL/tok |
|---|---|---|---|---|---|
| RTN INT4 g64 | 4.25 | +0.451 | [0.329, 0.58] | 88.68 | 0.0908 |
| RTN INT3 g64 | 3.25 | +1.911 | [1.619, 2.229] | 76.73 | 0.3452 |
| GPTQ INT3 asym g64 d0.01 (selected) | 3.297 | +0.621 | [0.491, 0.77] | 85.59 | 0.1419 |
| GPTQ INT3 asym g64 d0.1 | 3.297 | +0.513 | [0.403, 0.644] | 85.87 | 0.131 |
| GPTQ INT3 asym g64 d1.0 | 3.297 | +0.642 | [0.512, 0.793] | 85.85 | 0.1391 |
| GPTQ INT3 sym g64 d0.01 (selected) | 3.25 | +1.429 | [1.14, 1.795] | 79.59 | 0.2694 |
| GPTQ INT3 sym g64 d0.1 | 3.25 | +1.154 | [0.944, 1.37] | 81.37 | 0.2166 |
| GPTQ INT3 sym g64 d1.0 | 3.25 | +1.303 | [1.096, 1.53] | 80.88 | 0.2338 |
| kron13_2b1b_gptq, f16 | 3.550 | +0.382 | [0.269, 0.49] | 86.91 | 0.1264 |
| kron13_2b1b_gptq, INT8 | 3.314 | +0.390 | [0.279, 0.495] | 86.8 | 0.1276 |
| kron13_2b1b_gptq, INT4 | 3.195 | +0.564 | [0.458, 0.673] | 85.39 | 0.1572 |

The GPTQ controls are not bit-for-bit deterministic across runs (v2 vs v3 asym d0.01: +0.621 vs +0.661).
This run crashed (out of memory, a driver bug since fixed) after these rows; the rows above are complete.

### Run-to-run spread of one configuration (`results/derived_stats.json`)

kron13_2b1b_gptq, three independent runs (round 6, round 8 v2, round 8 v3), same code path and settings:
f16 +0.56 / +0.382 / +0.476 (sd 0.089); INT8 +0.564 / +0.39 / +0.478 (mean 0.477, sd 0.087);
INT4 +0.708 / +0.564 / +0.533 (sd 0.093). The 2-bit per-layer stage varies more: kron13_2bit_gptq f16 gave
+3.132 in round 6 and +1.688 in round 7 before end-to-end training.

## Round 7 — end-to-end KL distillation (single run; `results/round7_3b_results.json`, `results/round7_e2e_log.json`)

kron13_2bit_gptq per-layer stage, then 400 steps of joint KL(p_FP16 ‖ p_student) on calibration text over all
continuous parameters (codes fixed), INT4 fake-quantization of the factors during training.

| row | bpw | ΔPPL | 95% CI |
|---|---|---|---|
| RTN INT4 g64 | 4.25 | +0.439 | [0.318, 0.564] |
| RTN INT3 g64 | 3.25 | +1.897 | [1.607, 2.216] |
| before end-to-end training, f16 | 2.533 | +1.688 | [1.213, 2.374] |
| after, f16 | 2.533 | +0.339 | [0.06, 0.615] |
| after, INT8 | 2.296 | +0.334 | [0.055, 0.603] |
| after, INT4 | 2.178 | **+0.301** | [0.009, 0.59] |

Calibration-monitor KL (8 held-back calibration sequences): 0.2097 at step 0, flat at about 0.13 from step 150
to 400. Agreement with FP16 after training (re-scored from saved checkpoints, `results/agreement_r5_r6_r7.json`):
INT4 top-1 79.98%, KL 0.2587/token, R²h 74.15% — versus RTN INT4 at about 88–89% top-1 and 0.091 KL. Perplexity
improved more than fidelity to FP16: the trained model scores the eval tokens better than its KL distance to FP16
would suggest, consistent with partly fitting the WikiText domain shared by calibration and eval text.

## Rounds 5 and 6 — Kronecker core, nearest vs GPTQ rounding (`results/round5_3b_results.json`, `results/round6_3b_results.json`)

| row | bpw | round 5 (nearest) ΔPPL [CI] | round 6 (GPTQ) ΔPPL [CI] |
|---|---|---|---|
| RTN INT4 g64 | 4.25 | +0.444 [0.318, 0.572] | +0.451 [0.33, 0.576] |
| RTN INT3 g64 | 3.25 | +1.892 [1.603, 2.211] | +1.902 [1.609, 2.213] |
| RTN INT2 g64 | 2.25 | +5641.974 | +5710.919 |
| kron13_1bit, f16 / INT8 / INT4 | 1.533 / 1.297 / 1.178 | +84.499 / +84.389 / +92.064 | — |
| gate2_1bit (gate 2-bit), f16 / INT8 / INT4 | 1.866 / 1.630 / 1.511 | +48.897 / +48.644 / +49.458 | — |
| kron13_2bit, f16 | 2.533 | +1.903 [1.586, 2.214] | +3.132 [1.066, 8.542] |
| kron13_2bit, INT8 | 2.296 | +1.896 [1.578, 2.205] | +2.922 [1.058, 7.596] |
| kron13_2bit, INT4 | 2.178 | +2.112 [1.769, 2.458] | +1.478 [1.133, 1.911] |
| kron13_2b1b, f16 | 3.550 | +0.732 [0.625, 0.844] | +0.56 [0.455, 0.663] |
| kron13_2b1b, INT8 | 3.314 | +0.74 [0.633, 0.854] | +0.564 [0.457, 0.668] |
| kron13_2b1b, INT4 | 3.195 | +0.831 [0.725, 0.943] | +0.708 [0.592, 0.819] |

Agreement with FP16 (re-scored, `results/agreement_r5_r6_r7.json`; reloaded checkpoints reproduce each run's
ΔPPL within 0.01):

| run | factors | bpw | ΔPPL | top-1 % | KL/tok | R²h % |
|---|---|---|---|---|---|---|
| R5 kron13_2bit | INT8 | 2.296 | +1.901 | 74.71 | 0.4157 | 67.51 |
| R5 kron13_2b1b | INT8 | 3.314 | +0.745 | 84.77 | 0.1873 | 85.05 |
| R6 kron13_2bit_gptq | INT8 | 2.296 | +2.924 | 76.0 | 0.4912 | 68.43 |
| R6 kron13_2bit_gptq | INT4 | 2.178 | +1.48 | 77.52 | 0.3408 | 71.36 |
| R6 kron13_2b1b_gptq | INT8 | 3.314 | +0.566 | 85.65 | 0.1507 | 87.48 |
| R6 kron13_2b1b_gptq | INT4 | 3.195 | +0.71 | 84.79 | 0.1695 | 85.72 |
| R7 kron13_2bit_gptq + e2e | INT4 | 2.178 | +0.308 | 79.98 | 0.2587 | 74.15 |

GPTQ at 2 bits raised top-1 and R²h but worsened KL and PPL at f16/INT8 with very wide CIs: typical tokens
improved while a few sequences degraded badly. Per-layer data trace this to the attention-sink layers (down-matrix
weight R² at layer 3 fell from 81.0% to −12.3%) and to experts with few calibration tokens (`hess_tokens_min` per
layer in `results/round6_3b_results.json`).

## Round 3 — 1B model, 5 of 24 layers, 1-bit residual (`results/round3_1b_results.json`)

`granite-3.1-1b-a400m-base`, layers 4, 8, 12, 16, 20 replaced, FP16 PPL 8.302; point estimates only (no CIs).

| variant | bpw | ΔPPL | top-1 % | mean open / closed R² % |
|---|---|---|---|---|
| base (kron13_1bit) | 1.677 | +0.503 | 80.31 | 56.83 / 51.97 |
| whitened Kronecker SVD init (wsvd) | 1.677 | +0.573 | 79.97 | 54.58 / 49.91 |
| per-expert rank redistribution (expalloc) | 1.677 | +0.587 | 80.41 | 57.85 / 53.0 |
| per-layer rank allocation (layeralloc) | 1.645 | +0.476 | 81.31 | 60.67 / 53.85 |
