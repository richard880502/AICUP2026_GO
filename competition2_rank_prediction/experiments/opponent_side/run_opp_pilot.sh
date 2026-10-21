#!/usr/bin/env bash
# Opponent-side dense features for a pilot subset (3 train shards, 2 val shards): one worker per shard.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --opponent --depth 3"
c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/opp_train --output experiments/opponent_side/features/feat96opp/train $A --device 0 --worker 0 --num-workers 2 > experiments/opponent_side/logs/opp_w0.log 2>&1 &
c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/opp_train --output experiments/opponent_side/features/feat96opp/train $A --device 1 --worker 1 --num-workers 2 > experiments/opponent_side/logs/opp_w1.log 2>&1 &
c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/opp_val --output experiments/opponent_side/features/feat96opp/val $A --device 0 --worker 0 --num-workers 2 > experiments/opponent_side/logs/opp_w2.log 2>&1 &
c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/opp_val --output experiments/opponent_side/features/feat96opp/val $A --device 1 --worker 1 --num-workers 2 > experiments/opponent_side/logs/opp_w3.log 2>&1 &
wait
