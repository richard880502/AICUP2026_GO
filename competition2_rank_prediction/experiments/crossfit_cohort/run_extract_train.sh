#!/usr/bin/env bash
# 2026-10-21: 2,000-player cross-fitted cohort (features/holdout3, folds in question_folds.csv).
# Extraction queue, 4 KataGo workers (2 per GPU), each on its own fixed shard subset (features/shards/*_w<w>):
#   1. opponent side of the remaining 12,211 training games (for an opponent-specific model)  -> features/feat96opp/train
#   2. holdout3 player side -> features/feat96/holdout3      3. holdout3 opponent side -> features/feat96opp/holdout3
#   4. holdout3 final-position result proxy (workers 0 and 1)  -> features/final_result/holdout3
# Training in parallel: d96_c{0,1} and resnet8c_c{0,1}, each excluding one rank-balanced half of the cohort.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
L=experiments/crossfit_cohort/logs
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --depth 3"
retry() {  # log cmd...
  local log=$1; shift
  for attempt in 1 2 3 4 5 6; do "$@" >> "$log" 2>&1 && return; echo "attempt $attempt failed" >> "$log"; sleep 60; done
}
worker() {
  local w=$1 d=$(($1 % 2))
  KATAGO_STDERR=$L/kg_w$w.log retry $L/ext_w$w.log c1/.venv/bin/python extract_rank_features.py \
    --shard-dir features/shards/oppfull_train_w$w --output features/feat96opp/train $A --opponent --device $d
  KATAGO_STDERR=$L/kg_w$w.log retry $L/ext_w$w.log c1/.venv/bin/python extract_rank_features.py \
    --shard-dir features/shards/holdout3_w$w --output features/feat96/holdout3 $A --device $d
  KATAGO_STDERR=$L/kg_w$w.log retry $L/ext_w$w.log c1/.venv/bin/python extract_rank_features.py \
    --shard-dir features/shards/holdout3_w$w --output features/feat96opp/holdout3 $A --opponent --device $d
}
for w in 0 1 2 3; do worker $w & done
for j in 0 1; do
  OMP_NUM_THREADS=2 c1/.venv/bin/python resnet_rank.py --device cuda:$j --epochs 8 --workers 4 --cosine --tag resnet8c_c$j \
    --exclude-players features/holdout3/exclude_c$j.txt > $L/resnet8c_c$j.log 2>&1 &
done
for j in 0 1; do
  OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py --train features/feat96/train --val features/feat96/val --seeds 3 \
    --epochs 40 --device cuda:$j --tag d96_c$j --exclude-players features/holdout3/exclude_c$j.txt > $L/d96_c$j.log 2>&1
done &
wait
for w in 0 1; do
  c1/.venv/bin/python scripts/extract_final_result.py --shard-dir features/shards/holdout3 --output features/final_result/holdout3 \
    --device $w --worker $w --num-workers 2 > $L/final_w$w.log 2>&1 &
done
wait
echo EXTRACT_TRAIN_DONE >> $L/done.log
