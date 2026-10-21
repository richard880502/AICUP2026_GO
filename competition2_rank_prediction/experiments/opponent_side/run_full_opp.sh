#!/usr/bin/env bash
# 2026-10-20: opponent-side features for the whole validation cohort (6,000 games not in the pilot) and the internal holdout.
# GPU 0 -> holdout, GPU 1 -> validation, 1 worker each (2 per GPU runs KataGo out of memory next to the other users jobs); a worker whose KataGo dies is restarted (finished shards are skipped).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
O=experiments/opponent_side
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --opponent --depth 3"
worker() {  # split shard_dir out device w
  for attempt in 1 2 3 4 5 6; do
    KATAGO_STDERR=$O/logs/full_$1_w$5.katago.log c1/.venv/bin/python extract_rank_features.py --shard-dir $2 --output $3 $A --device $4 --worker $5 --num-workers 1 \
      >> $O/logs/full_$1_w$5.log 2>&1 && break
    echo "attempt $attempt failed, restarting in 60s" >> $O/logs/full_$1_w$5.log; sleep 60
  done
}
worker holdout features/shards/holdout $O/features/feat96opp/holdout 0 0 &
worker val features/shards/oppfull_val $O/features/feat96opp/val 1 0 &
wait
echo FULL_DONE >> $O/logs/full_done.log
