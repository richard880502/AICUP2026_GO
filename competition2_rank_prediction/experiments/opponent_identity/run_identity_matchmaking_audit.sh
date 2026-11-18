#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PY=${PY:-competition2_rank_prediction/c1/.venv/bin/python}
RAW_CACHE=${RAW_CACHE:-production/open_set_v2_20261111/raw_cache}
OUT=${OUT:-runtime/competition2_research/experiments/identity_matchmaking_graph}
OFFICIAL_SCORES=${OFFICIAL_SCORES:-/tmp2/richard/claude_scratch/opp_probe/test_opp_scores.npz}
CONTROLS=${CONTROLS:-/tmp2/richard/claude_scratch/opp_probe/controls.npz}

export PYTHONPATH=".:competition1_player_identification"

"$PY" -m   competition2_rank_prediction.experiments.opponent_identity.audit_identity_matchmaking_graph   --raw-cache "$RAW_CACHE"   --output "$OUT"   --official-scores "$OFFICIAL_SCORES"   --controls "$CONTROLS"
