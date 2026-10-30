#!/usr/bin/env bash
# For a fresh independent clone under an exclusively new task root.
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
test ! -e "$TASK_ROOT/venv" || { echo 'Refusing to reuse an environment. Choose a new task root.' >&2; exit 1; }
mkdir -p "$TMPDIR"
python3 -B -c 'import sys,platform; assert sys.version_info[:2]==(3,14) and platform.machine()=="x86_64", "Verified setup requires Linux x86_64 Python3.14"'
python3 -B -m venv "$TASK_ROOT/venv"
"$TASK_ROOT/venv/bin/python" -B -m pip install 'https://download-r2.pytorch.org/whl/cu126/torch-2.14.1%2Bcu126-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=eb7564fdaccd2920ed78b4e3c08ca4a180fcaf5d4b3ce3068b1968bce4f5834b'
"$TASK_ROOT/venv/bin/python" -B -m pip install -r "$VERIFY_PREFIX/requirements_verified.txt"
