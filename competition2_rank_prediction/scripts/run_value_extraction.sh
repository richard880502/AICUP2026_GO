#!/usr/bin/env bash
# Resumable value-feature extraction (NUM_WORKERS KataGo processes on DEVICE); val first, then train.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NUM_WORKERS=${NUM_WORKERS:-2}
DEVICE=${DEVICE:-0}
for ((w = 0; w < NUM_WORKERS; w++)); do
  (
    for split in val train; do
      c1/.venv/bin/python extract_value_features.py --shard-dir "features/shards/$split" \
        --output "features/feat_value/$split" --device "$DEVICE" --worker "$w" --num-workers "$NUM_WORKERS"
    done
  ) > "logs/value_w${w}.log" 2>&1 &
done
wait
