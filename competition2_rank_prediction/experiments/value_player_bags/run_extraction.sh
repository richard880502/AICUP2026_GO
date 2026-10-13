#!/usr/bin/env bash
# Extracts normal-KataGo value and ownership features for the dense bag cohort.
# The worker split is deterministic and each completed feature shard is skipped.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

workers=${NUM_WORKERS:-8}
device=${DEVICE:-0}
output=${OUTPUT:-experiments/value_player_bags/features}

mkdir -p "$output"

for ((worker = 0; worker < workers; worker++)); do
  c1/.venv/bin/python extract_value_features.py \
    --shard-dir experiments/dense_player_bags/shards \
    --output "$output" \
    --device "$device" \
    --worker "$worker" \
    --num-workers "$workers" \
    --depth 3 \
    --samples-per-game 32 \
    > "$output/extract_w${worker}.log" 2>&1 &
done

wait
