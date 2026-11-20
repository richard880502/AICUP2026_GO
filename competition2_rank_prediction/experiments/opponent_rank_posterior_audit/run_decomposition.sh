#!/usr/bin/env bash
# Read-only diagnostic on existing holdout3 v47 cache; same branch as P0/P1.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-runtime/competition2_research/c1/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python}"
fi
GPU="${GPU:-1}"
DEVICE="${DEVICE:-cuda:0}"
STATE="${STATE:-/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt}"
LABELS="${LABELS:-runtime/competition2_research/features/holdout3/question_folds.csv}"
OUT="${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit}"
if [[ ! -f "$STATE" ]]; then
  echo "Missing frozen v47 cache: $STATE" >&2
  exit 2
fi
if [[ ! -f "$LABELS" ]]; then
  echo "Missing holdout3 labels: $LABELS" >&2
  exit 2
fi
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m \
  competition2_rank_prediction.experiments.opponent_rank_posterior_audit.decompose_v47 \
  --state "$STATE" --labels "$LABELS" \
  --output "$OUT/v47_error_decomposition.json" \
  --device "$DEVICE" --steps 300 --bootstrap 3000
printf 'Diagnostic report: %s/v47_error_decomposition.json\n' "$OUT"
