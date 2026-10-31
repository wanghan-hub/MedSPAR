#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MODE=${1:-train}
case "$MODE" in train|eval) ;; *) echo 'Use train or eval' >&2; exit 2;; esac
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$MODE" == train ]]; then bash scripts/train_stage1_tc_mve.sh "$@"; fi
bash scripts/eval_stage1_tc_mve.sh
if [[ "$MODE" == train ]]; then bash scripts/train_main.sh "$@"; fi
bash scripts/eval_stage2_medspar.sh --experiment main
# Stage0 is the independent visual preparation branch, not Stage1 initialization.
if [[ "$MODE" == train ]]; then bash scripts/train_stage0_medsam.sh "$@"; fi
bash scripts/eval_stage0_medsam.sh
bash scripts/run_all_ablations.sh "$MODE" "$@"
