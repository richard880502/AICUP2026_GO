#!/usr/bin/env bash
# Dense value-feature pilot (96 decisions, normal model, transition tokens): same 3 train + 2 val shards as the opponent pilot.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="--samples-per-game 96"
c1/.venv/bin/python extract_value_features.py --shard-dir features/shards/opp_train --output experiments/value_pilot_dense/features/feat96val/train $A --device 0 --worker 0 --num-workers 2 > logs/v96_w0.log 2>&1 &
c1/.venv/bin/python extract_value_features.py --shard-dir features/shards/opp_train --output experiments/value_pilot_dense/features/feat96val/train $A --device 1 --worker 1 --num-workers 2 > logs/v96_w1.log 2>&1 &
c1/.venv/bin/python extract_value_features.py --shard-dir features/shards/opp_val --output experiments/value_pilot_dense/features/feat96val/val $A --device 0 --worker 0 --num-workers 2 > logs/v96_w2.log 2>&1 &
c1/.venv/bin/python extract_value_features.py --shard-dir features/shards/opp_val --output experiments/value_pilot_dense/features/feat96val/val $A --device 1 --worker 1 --num-workers 2 > logs/v96_w3.log 2>&1 &
wait
