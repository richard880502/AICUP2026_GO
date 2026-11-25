#!/usr/bin/env bash
# Reuses v4 controls.npz and v47 cross-fitted output, never refits or changes production.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/competition1_player_identification${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-runtime/competition2_research/c1/.venv/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python}"
fi
PROBE="${PROBE:-/tmp2/richard/claude_scratch/opp_probe}"
OUT="${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit}"
CONTROLS="${CONTROLS:-$PROBE/controls.npz}"
PRESENT="${PRESENT:-$PROBE/present.csv}"
H3_V4="${H3_V4:-$PROBE/h3_sims.pt}"
if [[ ! -f "$H3_V4" && -f "production/rank_opponent_identity_20261114/h3_sims.pt" ]]; then
  H3_V4="production/rank_opponent_identity_20261114/h3_sims.pt"
fi
DECOMPOSITION="${DECOMPOSITION:-$OUT/v47_error_decomposition.npz}"
for file in "$CONTROLS" "$PRESENT"; do
  if [[ ! -f "$file" ]]; then
    echo "Missing $file; rebuild with opponent_identity/link_test.py or supply CONTROLS/PRESENT." >&2
    exit 1
  fi
done
mkdir -p "$OUT"
EXTRA=()
if [[ -f "$H3_V4" && -f "$DECOMPOSITION" ]]; then
  EXTRA=(--h3-v4 "$H3_V4" --decomposition "$DECOMPOSITION")
  echo 'Optional v47 proxy diagnostic: enabled (v4-to-masked-gallery domain shift flagged).'
else
  echo 'Optional v47 proxy diagnostic: skipped (missing h3_sims.pt or decomposition npz).'
fi
"$PY" -m competition2_rank_prediction.experiments.opponent_rank_posterior_audit.known_unknown_audit \
  --controls "$CONTROLS" --present-csv "$PRESENT" \
  --output "$OUT/known_unknown_audit.json" "${EXTRA[@]}"
echo "Report: $OUT/known_unknown_audit.json"