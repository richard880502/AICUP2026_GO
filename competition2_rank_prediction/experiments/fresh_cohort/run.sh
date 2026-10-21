#!/usr/bin/env bash
# 2026-10-20 fresh player-disjoint cohort (features/holdout2, 400 new players) to confirm the opponent-side member.
# Models *_h2 exclude BOTH the old holdout and the new cohort (features/holdout2/exclude_h2.txt); they live in results/ because
# predict_test.py loads members from there.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
L=experiments/fresh_cohort/logs
EX=features/holdout2/exclude_h2.txt
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --depth 3"
OMP_NUM_THREADS=2 c1/.venv/bin/python resnet_rank.py --device cuda:0 --epochs 8 --workers 6 --cosine --tag resnet8c_h2 \
  --exclude-players $EX > $L/resnet8c_h2.log 2>&1 &
OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py --train features/feat96/train --val features/feat96/val --seeds 3 --epochs 40 \
  --device cuda:1 --tag d96_h2 --exclude-players $EX > $L/d96_h2.log 2>&1 &
for d in 0 1; do
  KATAGO_STDERR=$L/player_w$d.katago.log c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/holdout2 \
    --output features/feat96/holdout2 $A --device $d --worker $d --num-workers 2 > $L/player_w$d.log 2>&1 &
  KATAGO_STDERR=$L/opp_w$d.katago.log c1/.venv/bin/python extract_rank_features.py --shard-dir features/shards/holdout2 \
    --output experiments/opponent_side/features/feat96opp/holdout2 $A --opponent --device $d --worker $d --num-workers 2 > $L/opp_w$d.log 2>&1 &
done
wait
echo RUN_DONE >> $L/done.log
