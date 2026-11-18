#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=${PY:-competition2_rank_prediction/c1/.venv/bin/python}
GPU=${GPU:-1}
DEVICE=${DEVICE:-cuda:0}
STATE=${STATE:-/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt}
OUT=${OUT:-runtime/competition2_research/experiments/hard_boundary_atlas}
EXTRA_NPZ=${EXTRA_NPZ:-}

export PYTHONPATH=".:competition1_player_identification"
export CUDA_VISIBLE_DEVICES="$GPU"

extra_args=()
if [[ -n "$EXTRA_NPZ" ]]; then
  IFS=':' read -r -a extra_files <<< "$EXTRA_NPZ"
  extra_args=(--extra-npz "${extra_files[@]}")
fi

"$PY" -m \
  competition2_rank_prediction.experiments.hard_boundary_atlas.build_features \
  --state "$STATE" \
  --device "$DEVICE" \
  --output "$OUT/feature_table.npz" \
  "${extra_args[@]}"

"$PY" -m \
  competition2_rank_prediction.experiments.hard_boundary_atlas.analyze \
  --table "$OUT/feature_table.npz" \
  --output "$OUT"

echo "Atlas complete:"
echo "  $OUT/report.html"
echo "  $OUT/report.json"
echo "  $OUT/feature_metrics.csv"
echo "  $OUT/figures/"
