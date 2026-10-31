#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
source scripts/experiment_defaults.sh
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
bash scripts/train_stage2_medspar.sh --experiment no_spatial_heatmap_loss "$@"
