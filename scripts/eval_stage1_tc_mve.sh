#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
source scripts/experiment_defaults.sh
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
python tools/run_experiments.py eval --stage stage1 "$@"
