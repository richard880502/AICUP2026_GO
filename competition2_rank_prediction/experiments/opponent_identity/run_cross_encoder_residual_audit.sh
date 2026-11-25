#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=${PY:-competition2_rank_prediction/c1/.venv/bin/python}
GPU=${GPU:-1}
SAMPLE=${SAMPLE:-/tmp2/richard/claude_scratch/opp_probe/opp_sample.csv}
WORK=${WORK:-/tmp2/richard/claude_scratch/opp_probe/cross_encoder_residual}
V4_TOKENS=${V4_TOKENS:-$WORK/v4_tokens}
V7_TOKENS=${V7_TOKENS:-$WORK/v7_rich}
OFFICIAL_SCORES=${OFFICIAL_SCORES:-/tmp2/richard/claude_scratch/opp_probe/test_opp_scores.npz}
OFFICIAL_V7=${OFFICIAL_V7:-/tmp2/richard/claude_scratch/opp_probe/test_opp_sims_v7.npy}
OUTPUT=${OUTPUT:-runtime/competition2_research/experiments/identity_matchmaking_graph/cross_encoder_report.json}

export PYTHONPATH=".:competition1_player_identification"
export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$WORK"

if [[ ! -f "$V4_TOKENS/COMPLETE" ]]; then
  echo "[1/3] extracting v4 local HumanSL tokens for opp_sample"
  "$PY" -m competition1_player_identification.experiments.trunk_style.extract     --games "$SAMPLE"     --output "$V4_TOKENS"     --samples 160     --local-only     --batch-games 8     --workers 8
else
  echo "[1/3] reusing $V4_TOKENS"
fi

if [[ ! -f "$V7_TOKENS/COMPLETE" ]]; then
  echo "[2/3] extracting v7 rich HumanSL tokens for opp_sample"
  "$PY" -m competition1_player_identification.experiments.trunk_style.extract_rich     run     --games "$SAMPLE"     --output "$V7_TOKENS"     --samples 160     --batch-games 8     --workers 8
else
  echo "[2/3] reusing $V7_TOKENS"
fi

echo "[3/3] running cross-encoder residual audit"
"$PY" -m   competition2_rank_prediction.experiments.opponent_identity.audit_cross_encoder_residual   --sample "$SAMPLE"   --v4-tokens "$V4_TOKENS"   --v7-tokens "$V7_TOKENS"   --official-scores "$OFFICIAL_SCORES"   --official-v7 "$OFFICIAL_V7"   --output "$OUTPUT"
