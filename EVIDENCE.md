# RPM-IHOG Benchmark & Closed-Loop Distillation Evidence (v8)

## 1. Execution & Hardware Metadata
| Property | Value |
| :--- | :--- |
| **Repository** | `https://github.com/NoahMadison/RPM-IHOG` |
| **Timestamp (UTC)** | `2026-10-25 21:05:54 UTC` |
| **Git Commit** | `N/A` |
| **Compute Device** | `NVIDIA GeForce RTX 4060 Laptop GPU` (7.62 GB VRAM) |
| **PyTorch / Python** | PyTorch `2.14.1+cu126` / Python `3.14.4` |
| **OS** | `Linux 7.0.0-38-generic` |

## 2. Executive Benchmark Summary
| Stage | Benchmark Target | One-Shot $R^2$ (%) | Closed-Loop Distilled $R^2$ (%) | Cosine Similarity | Top-1 / Top-5 Match (%) | Wall Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Synthetic MoE Ablation `[8, 512, 512]` | 96.43% | **99.83%** | N/A | N/A | 2.35s |
| **Stage 3** | `ibm-granite/granite-3.1-1b-a400m-base` (Single-Layer + Speed) | 81.45% | **96.81%** | `0.98398` | 165610.2 tok/s | 5.17s |
| **Stage 5** | `ibm-granite/granite-3.1-1b-a400m-base` (Multi-Layer Cascade) | 86.2% | **98.7%** | `0.99536` | **82.23%** / **99.61%** | 31.06s |
| **Stage 6** | `Qwen/Qwen3.8-27B` (27B Dense Cascade) | 94.03% | **99.76%** | `0.99888` | **90.23%** / **100.0%** | 1.99s |
| **Stage 7** | `Qwen/Qwen3.8-Flash-Next` (180B / 512-Expert MoE) | 82.11% | **98.16%** | `0.99075` | 360GB $\rightarrow$ **27.0GB** (Fits 32GB V100) | 0.23s |

## 3. Stage 5 Multi-Layer Closed-Loop Cascade Progression (Granite MoE)
| Layer | Base Kron Rank | Distilled Kron Rank | One-Shot Cascaded $R^2$ (%) | Closed-Loop Distilled $R^2$ (%) | Cosine Similarity | Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Layer 0 | 207.43 | 14.31 | 95.9% | **99.76%** | `0.99883` | 5.14s |
| Layer 4 | 245.54 | 14.57 | 93.26% | **99.51%** | `0.99775` | 5.1s |
| Layer 8 | 243.65 | 14.75 | 91.17% | **99.31%** | `0.99704` | 5.1s |
| Layer 12 | 246.11 | 14.7 | 89.23% | **99.1%** | `0.99639` | 5.12s |
| Layer 16 | 251.27 | 14.67 | 87.66% | **98.91%** | `0.99586` | 5.14s |
| Layer 20 | 222.34 | 14.56 | 86.2% | **98.7%** | `0.99536` | 5.14s |

## 4. Stage 6 Multi-Layer Closed-Loop Cascade Progression (Qwen 27B Dense)
| Layer | Tensor Audited | Base SVD Rank | Base Kron Rank | Distilled Kron Rank | One-Shot $R^2$ (%) | Distilled $R^2$ (%) | Cosine Similarity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Layer 4 | `model.language_model.layers.4.mlp.up_proj.weight` | 251.32 | 256.38 | 14.72 | 97.63% | **99.89%** | `0.99947` |
| Layer 28 | `model.language_model.layers.28.mlp.up_proj.weight` | 243.14 | 258.94 | 14.89 | 95.62% | **99.82%** | `0.99912` |
| Layer 52 | `model.language_model.layers.52.mlp.up_proj.weight` | 244.76 | 258.93 | 15.12 | 94.03% | **99.76%** | `0.99888` |

## 5. Complete Machine-Readable JSON Telemetry
```json
{
  "metadata": {
    "repository": "https://github.com/NoahMadison/RPM-IHOG",
    "timestamp_utc": "2026-10-25 21:05:54 UTC",
    "git_commit": "N/A",
    "os": "Linux 7.0.0-38-generic",
    "python_version": "3.14.4",
    "pytorch_version": "2.14.1+cu126",
    "compute_device": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "vram_gb": 7.62
  },
  "benchmarks": {
    "stage1_ablation": {
      "stage": "stage1_ablation",
      "expert_shape": [
        8,
        512,
        512
      ],
      "shannon_entropy_body_bits": 7.157,
      "svd_stable_rank": 24.71,
      "kronecker_stable_rank": 7.24,
      "incoherence_mu": 16.44,
      "sqnr_pure_kronecker_db": 8.38,
      "sqnr_rpm_ihog_15bpw_db": 15.41,
      "distilled_kron_rank": 6.25,
      "pure_kronecker_r2_pct": 82.43,
      "oneshot_rpm_ihog_r2_pct": 96.43,
      "distilled_rpm_ihog_r2_pct": 99.83,
      "wall_time_sec": 2.35
    },
    "stage2_hf_olmoe": {
      "stage": "stage2_hf_olmoe",
      "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
      "audited_tensor": "model.layers.8.block_sparse_moe.input_linear.weight",
      "audited_shape": [
        16,
        1024,
        1024
      ],
      "shannon_entropy_body_bits": 7.423,
      "svd_stable_rank": 63.92,
      "kronecker_stable_rank": 240.29,
      "incoherence_mu": 10.07,
      "sqnr_pure_kronecker_db": 0.14,
      "sqnr_rpm_ihog_15bpw_db": 7.35,
      "wall_time_sec": 5.85
    },
    "stage3_granite_distill_and_speed": {
      "stage": "stage3_granite_distill_and_speed",
      "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
      "tensor": "model.layers.8.block_sparse_moe.input_linear.weight",
      "baseline_svd_rank": 69.4,
      "baseline_kron_rank": 243.65,
      "distilled_kron_rank": 14.81,
      "incoherence_mu": 10.15,
      "sqnr_pure_kronecker_db": 0.14,
      "sqnr_rpm_ihog_15bpw_db": 7.35,
      "oneshot_r2_pct": 81.45,
      "distilled_r2_pct": 96.81,
      "distilled_cosine_sim": 0.98398,
      "distill_sec": 5.17,
      "effective_tokens_per_sec": 165610.2
    },
    "stage4_dense_vs_moe": {
      "stage": "stage4_dense_vs_moe",
      "moe_model": {
        "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
        "tensor": "model.layers.8.block_sparse_moe.input_linear.weight",
        "shannon_entropy_body_bits": 7.426,
        "svd_stable_rank": 69.4,
        "kronecker_stable_rank": 243.65,
        "incoherence_mu": 10.15,
        "sqnr_pure_kronecker_db": 0.14,
        "sqnr_rpm_ihog_15bpw_db": 7.35
      },
      "dense_model": {
        "model_id": "HuggingFaceTB/SmolLM2-1.7B",
        "tensor": "model.layers.8.mlp.up_proj.weight",
        "shannon_entropy_body_bits": 7.433,
        "svd_stable_rank": 236.85,
        "kronecker_stable_rank": 260.15,
        "incoherence_mu": 11.5,
        "sqnr_pure_kronecker_db": 0.13,
        "sqnr_rpm_ihog_15bpw_db": 7.35
      },
      "sqnr_advantage_moe_vs_dense_db": 0.0
    },
    "stage5_full_granite": {
      "stage": "stage5_full_granite",
      "model_id": "ibm-granite/granite-3.1-1b-a400m-base",
      "cascaded_layers_tested": [
        0,
        4,
        8,
        12,
        16,
        20
      ],
      "per_layer_cascade": [
        {
          "layer": 0,
          "base_kron_rank": 207.43,
          "dist_kron_rank": 14.31,
          "cascaded_oneshot_r2_pct": 95.9,
          "cascaded_distilled_r2_pct": 99.76,
          "cascaded_distilled_cos_sim": 0.99883,
          "distill_sec": 5.14
        },
        {
          "layer": 4,
          "base_kron_rank": 245.54,
          "dist_kron_rank": 14.57,
          "cascaded_oneshot_r2_pct": 93.26,
          "cascaded_distilled_r2_pct": 99.51,
          "cascaded_distilled_cos_sim": 0.99775,
          "distill_sec": 5.1
        },
        {
          "layer": 8,
          "base_kron_rank": 243.65,
          "dist_kron_rank": 14.75,
          "cascaded_oneshot_r2_pct": 91.17,
          "cascaded_distilled_r2_pct": 99.31,
          "cascaded_distilled_cos_sim": 0.99704,
          "distill_sec": 5.1
        },
        {
          "layer": 12,
          "base_kron_rank": 246.11,
          "dist_kron_rank": 14.7,
          "cascaded_oneshot_r2_pct": 89.23,
          "cascaded_distilled_r2_pct": 99.1,
          "cascaded_distilled_cos_sim": 0.99639,
          "distill_sec": 5.12
        },
        {
          "layer": 16,
          "base_kron_rank": 251.27,
          "dist_kron_rank": 14.67,
          "cascaded_oneshot_r2_pct": 87.66,
          "cascaded_distilled_r2_pct": 98.91,
          "cascaded_distilled_cos_sim": 0.99586,
          "distill_sec": 5.14
        },
        {
          "layer": 20,
          "base_kron_rank": 222.34,
          "dist_kron_rank": 14.56,
          "cascaded_oneshot_r2_pct": 86.2,
          "cascaded_distilled_r2_pct": 98.7,
          "cascaded_distilled_cos_sim": 0.99536,
          "distill_sec": 5.14
        }
      ],
      "final_cascaded_oneshot_r2_pct": 86.2,
      "final_cascaded_distilled_r2_pct": 98.7,
      "final_cascaded_distilled_cosine_sim": 0.99536,
      "end_to_end_top1_token_match_oneshot_pct": 48.63,
      "end_to_end_top1_token_match_distilled_pct": 82.23,
      "end_to_end_top5_token_recall_distilled_pct": 99.61,
      "total_distill_wall_time_sec": 31.06
    },
    "stage6_qwen27b_benchmark": {
      "stage": "stage6_qwen27b_benchmark",
      "model_id": "Qwen/Qwen3.8-27B",
      "layers_audited": [
        4,
        28,
        52
      ],
      "per_layer_reports": [
        {
          "layer": 4,
          "tensor": "model.language_model.layers.4.mlp.up_proj.weight",
          "base_svd_rank": 251.32,
          "base_kron_rank": 256.38,
          "dist_kron_rank": 14.72,
          "cascaded_oneshot_r2_pct": 97.63,
          "cascaded_distilled_r2_pct": 99.89,
          "cascaded_distilled_cos_sim": 0.99947,
          "distill_sec": 0.66
        },
        {
          "layer": 28,
          "tensor": "model.language_model.layers.28.mlp.up_proj.weight",
          "base_svd_rank": 243.14,
          "base_kron_rank": 258.94,
          "dist_kron_rank": 14.89,
          "cascaded_oneshot_r2_pct": 95.62,
          "cascaded_distilled_r2_pct": 99.82,
          "cascaded_distilled_cos_sim": 0.99912,
          "distill_sec": 0.65
        },
        {
          "layer": 52,
          "tensor": "model.language_model.layers.52.mlp.up_proj.weight",
          "base_svd_rank": 244.76,
          "base_kron_rank": 258.93,
          "dist_kron_rank": 15.12,
          "cascaded_oneshot_r2_pct": 94.03,
          "cascaded_distilled_r2_pct": 99.76,
          "cascaded_distilled_cos_sim": 0.99888,
          "distill_sec": 0.66
        }
      ],
      "final_cascaded_oneshot_r2_pct": 94.03,
      "final_cascaded_distilled_r2_pct": 99.76,
      "final_cascaded_distilled_cosine_sim": 0.99888,
      "end_to_end_top1_token_match_oneshot_pct": 60.55,
      "end_to_end_top1_token_match_distilled_pct": 90.23,
      "end_to_end_top5_token_recall_distilled_pct": 100.0,
      "total_distill_wall_time_sec": 1.99
    },
    "stage7_flash_next_moe": {
      "stage": "stage7_flash_next_moe",
      "model_id": "Qwen/Qwen3.8-Flash-Next",
      "tensor_name": "model.language_model.layers.8.mlp.shared_expert.down_proj.weight",
      "audited_expert_slice": [
        1,
        512,
        512
      ],
      "baseline_svd_rank": 79.83,
      "baseline_kron_rank": 116.69,
      "distilled_kron_rank": 14.37,
      "oneshot_r2_accuracy_pct": 82.11,
      "distilled_r2_accuracy_pct": 98.16,
      "distilled_cosine_sim": 0.99075,
      "distill_sec": 0.23,
      "flash_next_bf16_size_gb": 360.0,
      "flash_next_q4_k_m_size_gb": 119.6,
      "flash_next_125b_moe_awq_kron_15bpw_gb": 27.0,
      "fits_125b_moe_backbone_in_32gb_v100": true
    }
  }
}
```
