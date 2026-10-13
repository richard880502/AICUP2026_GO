#!/usr/bin/env bash
# Dense extraction: up to 96 decisions per game, KataGo normal model + HumanSL rank_5k / rank_1d / rank_5d.
# 4 workers on 2 GPUs (worker w uses GPU w%2); holdout first, then validation, then train. Resumable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NW=${NUM_WORKERS:-4}
for ((w = 0; w < NW; w++)); do
  (
    for split in holdout val train; do
      c1/.venv/bin/python extract_rank_features.py --shard-dir "features/shards/$split" --output "features/feat96/$split" \
        --samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --device $((w % 2)) --worker "$w" --num-workers "$NW" --depth 3
    done
  ) > "logs/ext96_w${w}.log" 2>&1 &
done
wait
