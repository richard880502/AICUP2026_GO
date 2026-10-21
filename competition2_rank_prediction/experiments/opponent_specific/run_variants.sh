#!/usr/bin/env bash
# Opponent-specific Transformer: P(player's rank | opponent's 96 decisions), trained on the opponent side of all training games
# (features/feat96opp/train), excluding the old holdout + holdout2 players like d96_h2 (development models, suffix _h2).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
O=experiments/opponent_specific
until [ "$(ls features/feat96opp/train/feat_*.npz 2>/dev/null | wc -l)" -ge 16 ]; do sleep 60; done
C="--train features/feat96opp/train --val features/feat96opp/val --seeds 3 --exclude-players features/holdout2/exclude_h2.txt --out $O/results"
OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py $C --device cuda:0 --init-from d96_h2 --train-only head --lr 1e-3 --epochs 20 --tag opp_head_h2 > $O/logs/head.log 2>&1 &
OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py $C --device cuda:1 --init-from d96_h2 --lr 1e-4 --epochs 20 --tag opp_full_h2 > $O/logs/full.log 2>&1 &
OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py $C --device cuda:0 --epochs 40 --tag opp_scratch_h2 > $O/logs/scratch.log 2>&1 &
wait
OMP_NUM_THREADS=8 c1/.venv/bin/python experiments/opponent_specific/select_on_val.py > $O/logs/select.log 2>&1
echo VARIANTS_DONE >> $O/logs/select.log
