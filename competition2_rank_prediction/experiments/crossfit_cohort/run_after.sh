#!/usr/bin/env bash
# waits for extraction/training, then runs the pre-registered evaluation once
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
L=experiments/crossfit_cohort/logs
until grep -q EXTRACT_TRAIN_DONE $L/done.log 2>/dev/null && [ "$(ls results/opp_scratch_c1_seed*.pt 2>/dev/null | wc -l)" -ge 3 ] \
      && [ "$(ls results/opp_scratch_c0_seed*.pt 2>/dev/null | wc -l)" -ge 3 ] && ! pgrep -f "tag opp_scratch_c" > /dev/null; do sleep 60; done
OMP_NUM_THREADS=8 c1/.venv/bin/python -u experiments/crossfit_cohort/evaluate.py > $L/evaluate.log 2>&1
echo EVAL_DONE >> $L/done.log
