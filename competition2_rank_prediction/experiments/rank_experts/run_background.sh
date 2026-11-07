#!/usr/bin/env bash
# Starts the fixed cached pilot and captures progress and the final report.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo"
mkdir -p production/logs
exec runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.rank_experts.run \
  --device cuda:1 --epochs 200 --bootstrap 5000 \
  > production/logs/rank_experts_20261106.log 2>&1
