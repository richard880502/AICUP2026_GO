#!/usr/bin/env bash
# C2 cross-fitted models: opponent-specific Transformer from scratch, one per half of the holdout3 cohort.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
for j in 0 1; do
  OMP_NUM_THREADS=4 c1/.venv/bin/python transformer_rank.py --train features/feat96opp/train --val features/feat96opp/val --seeds 3 \
    --epochs 40 --device cuda:1 --tag opp_scratch_c$j --exclude-players features/holdout3/exclude_c$j.txt \
    > experiments/crossfit_cohort/logs/opp_scratch_c$j.log 2>&1 &
done
wait
