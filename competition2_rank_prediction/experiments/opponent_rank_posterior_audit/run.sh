#!/usr/bin/env bash
# Run from ANY directory; no new model fitting, frozen v4 controls only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PY:-runtime/competition2_research/c1/.venv/bin/python}"
# The historical scripts typically use the research root as cwd; prefer its venv.
if [[ ! -x "$PY" ]]; then
  PY="${PY_FALLBACK:-python}"
fi
GPU="${GPU:-1}"  # GPU 0 may be shared with another service.
PROBE="${PROBE:-/tmp2/richard/claude_scratch/opp_probe}"
OUT="${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit}"
mkdir -p "$PROBE" "$OUT"
if [[ ! -f "$PROBE/present.csv" ]]; then
  echo 'Generating *only* present.csv from unused games; not loading official tests.'
  "$PY" -c 'from pathlib import Path; from competition2_rank_prediction.experiments.opponent_identity import link_inputs as l; Path(l.OUT).mkdir(parents=True,exist_ok=True); l.write_present()'
  if [[ "$PROBE" != "/tmp2/richard/claude_scratch/opp_probe" ]]; then
    echo "Custom PROBE requires existing present.csv in that directory" >&2
    exit 1
  fi
fi
if ! ls "$PROBE"/present/shard_*.npz >/dev/null 2>&1; then
  echo 'Extracting missing frozen HumanSL trunk tokens (no training).'
fi
# Resumable: extraction skips shards already on disk. Exact record-id checks
# in score_present.py fail if this cache is incomplete or mismatched.
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m competition1_player_identification.experiments.trunk_style.extract \
  --games "$PROBE/present.csv" --output "$PROBE/present" \
  --samples 160 --local-only --workers "${WORKERS:-4}" --batch-games "${BATCH_GAMES:-4}"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m \
  competition2_rank_prediction.experiments.opponent_rank_posterior_audit.score_present \
  --present-csv "$PROBE/present.csv" --token-dir "$PROBE/present" \
  --output "$OUT/present_rank_scores.npz"
"$PY" -m competition2_rank_prediction.experiments.opponent_rank_posterior_audit.audit \
  --scores "$OUT/present_rank_scores.npz" --output "$OUT/report.json"
echo "Full report: $OUT/report.json"
