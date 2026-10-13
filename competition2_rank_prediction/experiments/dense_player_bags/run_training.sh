#!/usr/bin/env bash
# Trains both dense player-bag controls without overwriting either result.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

device=${DEVICE:-cpu}
threads=${CPU_THREADS:-4}
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"

c1/.venv/bin/python -m experiments.dense_player_bags.train_bags \
  --architecture deepsets \
  --device "$device" \
  --output experiments/dense_player_bags/results/deepsets

c1/.venv/bin/python -m experiments.dense_player_bags.train_bags \
  --architecture transformer \
  --device "$device" \
  --output experiments/dense_player_bags/results/transformer
