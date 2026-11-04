#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "$0")/env.sh"
exec "$TASK_ROOT/venv/bin/python" -B "$VERIFY_PREFIX/run_verification.py" "$@"
