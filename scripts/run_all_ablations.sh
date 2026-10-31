#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
source scripts/experiment_defaults.sh
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
MODE=${1:-train}
case "$MODE" in train|eval) ;; *) echo 'Use train or eval' >&2; exit 2;; esac
if [[ $# -gt 0 ]]; then shift; fi
for EXP in no_rq no_semantic box_only no_spatial_heatmap_loss medsam_vision coarse_mask_only semantic_only; do
  if [[ "$MODE" == train ]]; then bash scripts/train_stage2_medspar.sh --experiment "$EXP" "$@"; fi
  bash scripts/eval_stage2_medspar.sh --experiment "$EXP"
done
