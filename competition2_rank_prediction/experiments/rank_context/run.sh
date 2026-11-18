#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=${PY:-competition2_rank_prediction/c1/.venv/bin/python}
GPU=${GPU:-0}
DEVICE=${DEVICE:-cuda:0}
STATE=${STATE:-/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt}
H3_OPP=${H3_OPP:-/tmp2/richard/claude_scratch/opp_probe/h3_opp.csv}
HOLDOUT_ROOT=${HOLDOUT_ROOT:-runtime/competition2_research/features/holdout3}
CONTEXT=${CONTEXT:-runtime/competition2_research/experiments/rank_context/holdout3_context.npz}

export PYTHONPATH=".:competition1_player_identification"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PY" -m competition2_rank_prediction.experiments.rank_context.build_context   --state "$STATE"   --h3-opp "$H3_OPP"   --holdout-root "$HOLDOUT_ROOT"   --output "$CONTEXT"

"$PY" -m competition2_rank_prediction.experiments.rank_context.run   --state "$STATE"   --context "$CONTEXT"   --device "$DEVICE"
