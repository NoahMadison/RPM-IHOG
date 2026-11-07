# Reproducing the reported rows

## Environment

- GPU: one Tesla V100-PCIE-32GB (NVIDIA driver 580.178.04). Rounds 6–8 ran with the board power limited to 200 W;
  round 5 and earlier ran at the default 250 W. The limit affects speed, not results.
- Python 3.14.4; `torch==2.14.1+cu126` (CUDA 12.6 wheels), `transformers==4.57.1`, `numpy==2.5.4`,
  `matplotlib==3.11.2` — see `requirements.txt`.
- Model weights: `ibm-granite/granite-3.1-3b-a800m-base` (and `granite-3.1-1b-a400m-base` for round 3) from the
  Hugging Face Hub. The runs used a pre-populated cache with `HF_HUB_OFFLINE=1`; drop that variable on a first run.
- GPU memory: about 20 GB allocated at the end of a round-8 configuration; not recorded for round 7's end-to-end stage.

## Data

`load_disjoint_calib_and_eval_tokens` in `scripts/rpm_ihog_15bpw_hadamard_distill.py` reads
`./rpm_ihog_15bpw_results/real_corpus.txt`. If the file is missing it downloads WikiText-2 `valid.txt` from the
`pytorch/examples` GitHub repository (and, **if that download fails, silently falls back to TinyShakespeare** — check
that the file starts with ` = Homarus gammarus = `). The file used for every run here had MD5
`d756d22ab2310e7e60c094bea5c5a8eb` and 1,121,681 bytes; it is not included in this folder. The text is tokenized
once; calibration = tokens [0, 12288) as 48 × 256, evaluation = tokens [13312, 19456) as 24 × 256.

## Commands

Run from `scripts/` (the modules import each other by file name). Every run refuses a non-empty `--out_dir`.
`CUDA_VISIBLE_DEVICES` selected the V100 on the original machine; adjust for yours. Runtimes are wall-clock on the
V100 (about 40 s per layer per configuration for the 3B).

| rows | command | runtime |
|---|---|---|
| Round 3 (1B, 5 layers) | `python rpm_ihog_round3.py --out_dir ./rpm_ihog_round3` (defaults: 1B model, layers 4 8 12 16 20, exps base wsvd expalloc layeralloc) | ≤ 12 min (bounded by file times; not logged) |
| Round 5 (Kronecker, nearest rounding) | `python rpm_ihog_round5.py --model_id ibm-granite/granite-3.1-3b-a800m-base --out_dir ./rpm_ihog_round5_3b_run1` | ~2 h |
| Round 6 damping probes | `python rpm_ihog_round6_probe.py --out_dir ./rpm_ihog_round6_probe_3b` and `python rpm_ihog_round6_probe.py --layers 12 20 28 --res 2bit 2b1b --damps 1 2 5 20 --out_dir ./rpm_ihog_round6_probe_3b_v2` | ≤ 12 min each (bounded by file times) |
| Round 6 (Kronecker + GPTQ) | `python rpm_ihog_round6.py --configs kron13_2bit_gptq kron13_2b1b_gptq --damp 1.0 --out_dir ./rpm_ihog_round6_3b_gptq_d1_v3` | ~72 min |
| Round 7 (end-to-end KL) | `python rpm_ihog_round7.py --configs kron13_2bit_gptq --out_dir ./rpm_ihog_round7_3b_2bit_gptq_qat4` | ~56 min |
| Agreement re-scoring of rounds 5–7 | `python rpm_ihog_eval_agree.py --out_json ./agree_r5_r6_r7.json --run R5_2bit=kron13_2bit:<r5dir>:ckpt_kron13_2bit_L{l}.pt R5_2b1b=kron13_2b1b:<r5dir>:ckpt_kron13_2b1b_L{l}.pt R6_2bit_gptq=kron13_2bit_gptq:<r6dir>:ckpt_kron13_2bit_gptq_L{l}.pt R6_2b1b_gptq=kron13_2b1b_gptq:<r6dir>:ckpt_kron13_2b1b_gptq_L{l}.pt R7_2bit_e2e=kron13_2bit_gptq:<r7dir>:ckpt_kron13_2bit_gptq_L{l}_e2e.pt` | ~4 min |
| Round 8 v2 (controls incl. GPTQ INT3 sym, replicate) | `python rpm_ihog_round8.py --out_dir ./rpm_ihog_round8_3b_v2` | ~58 min until it stopped (see below) |
| Round 8 v3 (headline comparison) | `python rpm_ihog_round8.py --controls asym --out_dir ./rpm_ihog_round8_3b_v3` | ~65 min for the reported rows |
| Round 8 v4 (2-bit + sink rule) | `python rpm_ihog_round8.py --configs kron13_2bit_gptq_sink lr_2bit_gptq --controls --spectral_layers --out_dir ./rpm_ihog_round8_3b_v4` | ~61 min |
| Round 8 spectral | `python rpm_ihog_round8_spectral.py --ckpt_dir ./rpm_ihog_round8_3b_v3 --out_dir ./rpm_ihog_round8_spectral` | ~7 min |

To reproduce only the headline rows: `python rpm_ihog_round8.py --configs kron13_2b1b_gptq lr_2b1b_gptq --controls asym
--spectral_layers --out_dir ./r8_headline` (about 1 h).

Analysis (CPU only, from this folder): `python scripts/analysis/paired_tests.py`, `python scripts/analysis/derived_stats.py`,
`python scripts/analysis/make_frontier.py` regenerate `results/paired_tests.json`, `results/derived_stats.json` and
`figures/frontier_round8.png` from the other files in `results/`.

## Which code produced which rows

All files in `scripts/` are byte-identical copies of the files on the machine that ran the experiments. Their
modification times predate the runs they are cited for, with one exception:

- `rpm_ihog_round8.py` is the version that ran **round 8 v4**. Round 8 v2 and v3 ran earlier versions of the same
  file. v2 ran before two driver fixes: (a) blocks of the previous configuration were not freed before the next one
  (v2 ran out of GPU memory in its second configuration; all rows it reports are complete), and (b) per-sequence
  NLLs and the in-run paired comparison were not yet recorded. v3 ran with those fixes but with an earlier
  `fit_grouped_gains_` (a joint covariance-weighted solve across column groups) that only affects the
  `kron0_2b1b_gg128_gptq` configuration; v3 crashed in that configuration after its reported rows were written,
  and that configuration is not reported. The module docstring still describes the earlier solve.
- Round 8 also contained a bug fix before v2: the INT3 control's fp16 group scales underflowed to 0 for near-constant
  groups in layers 0–3 (first attempt, not reported). The copy here contains the fix.

## Nondeterminism

Training uses CUDA kernels that are not bit-reproducible. Identical configurations differed by up to 0.18 ΔPPL across
runs (kron13_2b1b_gptq INT8: +0.564, +0.39, +0.478), and the training-free GPTQ INT3 control by 0.04. The FP16
baseline itself varied between 6.928 and 6.935. Compare rows within one run, or average several runs.
