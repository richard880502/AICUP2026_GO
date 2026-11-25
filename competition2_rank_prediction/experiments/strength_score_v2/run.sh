#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=${PY:-competition2_rank_prediction/c1/.venv/bin/python}
DEVICE=${DEVICE:-cuda:0}
GPU=${GPU:-0}
EXP=runtime/competition2_research/experiments/strength_score_v2
H3=runtime/competition2_research/features/holdout3
OPP_STATE=${OPP_STATE:-/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt}

export PYTHONPATH=".:competition1_player_identification"

"$PY" -m competition2_rank_prediction.experiments.strength_score_v2.prepare_cache   --output "$EXP/train_cache"

"$PY" -m competition2_rank_prediction.experiments.strength_score_v2.prepare_holdout3   --holdout-root "$H3"   --output "$EXP/holdout3_own.csv"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m   competition1_player_identification.experiments.trunk_style.extract_rich   run   --games "$EXP/holdout3_own.csv"   --output "$EXP/holdout3_rich"   --samples 160   --batch-games 8   --workers 8

"$PY" -m competition2_rank_prediction.experiments.strength_score_v2.prepare_eval_cache   --games "$EXP/holdout3_own.csv"   --rich-dir "$EXP/holdout3_rich"   --output "$EXP/holdout3_cache"

for fold in 0 1; do
  "$PY" -m competition2_rank_prediction.experiments.strength_score_v2.train     --fold "$fold"     --cache "$EXP/train_cache"     --eval-cache "$EXP/holdout3_cache"     --exclude-players "$H3/exclude_c${fold}.txt"     --device "$DEVICE"
done

ARGS=(--device "$DEVICE")
if [[ -f "$OPP_STATE" ]]; then
  ARGS+=(--opponent-state "$OPP_STATE")
else
  echo "warning: $OPP_STATE not found; reporting Strength Score only" >&2
fi

"$PY" -m competition2_rank_prediction.experiments.strength_score_v2.evaluate_crossfit   "${ARGS[@]}"
