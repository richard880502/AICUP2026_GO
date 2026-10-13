#!/usr/bin/env bash
# Extracts 96-decision KataGo/HumanSL features for the targeted bag cohort.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

workers=${NUM_WORKERS:-12}
for ((worker = 0; worker < workers; worker++)); do
  c1/.venv/bin/python extract_rank_features.py \
    --shard-dir experiments/dense_player_bags/shards \
    --output experiments/dense_player_bags/features \
    --samples-per-game 96 \
    --profiles rank_5k,rank_1d,rank_5d \
    --device "$((worker % 2))" \
    --worker "$worker" \
    --num-workers "$workers" \
    --depth 3 \
    > "experiments/dense_player_bags/extract_w${worker}.log" 2>&1 &
done
wait
