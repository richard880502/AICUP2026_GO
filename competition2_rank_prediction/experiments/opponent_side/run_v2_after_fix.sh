#!/usr/bin/env bash
# 2026-10-20 re-run after fixing --opponent: wait for the 5 corrected shards, rebuild the pilot sets, train 3 variants x 3 seeds,
# then rerun the opponent part of experiments/a1d_boundary/run.py.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
O=experiments/opponent_side
until [ "$(ls $O/features/feat96opp/train/feat_*.npz $O/features/feat96opp/val/feat_*.npz 2>/dev/null | wc -l)" -ge 5 ] \
      && ! pgrep -f "extract_rank_features.py --shard-dir features/shards/opp" > /dev/null; do sleep 30; done
rm -rf $O/features/pilot_opp
PYTHONPATH=. c1/.venv/bin/python $O/build_opp_pilot.py > $O/logs/v2_build.log 2>&1 || exit 1
mkdir -p $O/results_v2
i=0
for v in player opponent both; do
  c1/.venv/bin/python transformer_rank.py --train $O/features/pilot_opp/$v/train --val $O/features/pilot_opp/$v/val \
    --seeds 3 --epochs 40 --device cuda:$((i % 2)) --tag opp_$v --out $O/results_v2 --questions-per-player 3 \
    > $O/logs/v2_opp_$v.log 2>&1 &
  i=$((i+1))
done
OMP_NUM_THREADS=6 c1/.venv/bin/python -u experiments/a1d_boundary/run.py > experiments/a1d_boundary/results/run_v3_opponent.log 2>&1
wait
echo ALL_DONE >> $O/logs/v2_build.log
