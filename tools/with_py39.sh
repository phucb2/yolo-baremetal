#!/usr/bin/env bash
# Run a command inside conda env `py39` (torch + ultralytics). Use: bash tools/with_py39.sh python tools/generate_layer_tests.py
set -euo pipefail
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py39
exec "$@"
