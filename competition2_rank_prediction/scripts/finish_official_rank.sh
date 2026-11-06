#!/usr/bin/env bash
# Complete official rank inference after a separate extraction process exits.
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 EXTRACTION_PID" >&2
  exit 2
fi

repo=/tmp2/richard/AICUP2026_GO
research="$repo/runtime/competition2_research"
production="$repo/production/rank_20261030"
tests="$repo/data/aicup2026_rank_prediction_tests_20261030/tests"
python_bin="$repo/runtime/competition1_research/.venv/bin/python"
extraction_pid=$1
cd "$research"

while kill -0 "$extraction_pid" 2>/dev/null; do
  sleep 30
done

shopt -s nullglob
input_shards=("$production"/shards/games_*.csv)
for feature_type in player opponent final; do
  feature_shards=("$production"/features/"$feature_type"/feat_*.npz)
  if [[ ${#feature_shards[@]} -ne ${#input_shards[@]} ]]; then
    echo "Incomplete $feature_type features: ${#feature_shards[@]} of ${#input_shards[@]} shards" >&2
    exit 1
  fi
done
if [[ ${#input_shards[@]} -eq 0 || ! -f "$production/moves.npz" ]]; then
  echo "Input shards or move tokens are missing" >&2
  exit 1
fi

"$python_bin" predict_test.py \
  --tests-dir "$tests" \
  --feat "$production/features/player" \
  --opp-feat "$production/features/opponent" \
  --final-result "$production/features/final" \
  --moves "$production/moves.npz" \
  --out "$production/submission_rank.csv" \
  --report "$production/prediction_report.csv" \
  --dump "$production/per_game_predictions.npz" \
  --expect 800 --strict --device cuda:1

"$python_bin" scripts/check_submission.py \
  "$production/submission_rank.csv" \
  --tests-dir "$tests" \
  --json "$production/submission_validation.json"
