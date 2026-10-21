#!/usr/bin/env bash
# Test-day dense extraction (run after scripts/test_to_games.py): the labelled player's decisions -> features/feat96/test and the
# opponent's decisions -> features/feat96opp/test, then the KataGo final-position features -> features/final_result/test
# (all three used by predict_test.py by default). One worker per pass per GPU
# (4 KataGo in total); set PER_GPU=1 to run one KataGo per GPU when other jobs leave little GPU memory (each needs ~1 GB).
# Resumable: finished shards are skipped, and a worker whose KataGo dies is restarted.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SHARDS=${SHARDS:-features/shards/test}
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --depth 3"
mkdir -p logs
run() {  # name output extra_flags device worker num_workers
  for attempt in 1 2 3 4 5 6; do
    KATAGO_STDERR=logs/test_$1_w$5.katago.log c1/.venv/bin/python extract_rank_features.py --shard-dir "$SHARDS" --output "$2" \
      $A $3 --device "$4" --worker "$5" --num-workers "$6" >> "logs/test_$1_w$5.log" 2>&1 && return
    echo "attempt $attempt failed, restarting in 60s" >> "logs/test_$1_w$5.log"; sleep 60
  done
}
if [ "${PER_GPU:-2}" = 1 ]; then   # GPU 0 player pass, GPU 1 opponent pass
  run player features/feat96/test "" 0 0 1 &
  run opponent features/feat96opp/test --opponent 1 0 1 &
else
  for d in 0 1; do
    run player features/feat96/test "" $d $d 2 &
    run opponent features/feat96opp/test --opponent $d $d 2 &
  done
fi
wait
# KataGo final-position result proxy for the shift residual (fast: one 64-visit query per game)
for d in 0 1; do
  c1/.venv/bin/python scripts/extract_final_result.py --shard-dir "$SHARDS" --output features/final_result/test --device $d \
    --worker $d --num-workers 2 >> logs/test_final_w$d.log 2>&1 &
done
wait
echo "final-result shards: $(ls features/final_result/test/feat_*.npz 2>/dev/null | wc -l)"
echo "player shards: $(ls features/feat96/test/feat_*.npz 2>/dev/null | wc -l), opponent shards: $(ls features/feat96opp/test/feat_*.npz 2>/dev/null | wc -l), input shards: $(ls $SHARDS/games_*.csv | wc -l)"
