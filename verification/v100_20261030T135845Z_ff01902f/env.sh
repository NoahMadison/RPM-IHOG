#!/usr/bin/env bash
# Source this file or invoke run.sh. HOME is intentionally preserved.
VERIFY_PREFIX="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="${VERIFY_PREFIX%/repo/verification/*}"
export VERIFY_PREFIX TASK_ROOT
export HF_HOME="$TASK_ROOT/cache/hf" HF_HUB_CACHE="$TASK_ROOT/cache/hf/hub"
export HF_DATASETS_CACHE="$TASK_ROOT/cache/datasets" TORCH_HOME="$TASK_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/extensions" XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export PIP_CACHE_DIR="$TASK_ROOT/cache/pip" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/triton" MPLCONFIGDIR="$TASK_ROOT/cache/mpl"
export TMPDIR="$TASK_ROOT/tmp" PYTHONDONTWRITEBYTECODE=1 WANDB_MODE=disabled
export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_XET=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=GPU-b1e26f14-9243-49b6-c5a7-51f7ce39672d
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PATH="$TASK_ROOT/venv/bin:$PATH"
export PYTHONPATH="$VERIFY_PREFIX/src${PYTHONPATH:+:$PYTHONPATH}"
