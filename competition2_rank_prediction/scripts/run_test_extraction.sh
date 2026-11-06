#!/usr/bin/env bash
# Test-day dense extraction (run after scripts/test_to_games.py): the labelled player's decisions -> features/feat96/test and the
# opponent's decisions -> features/feat96opp/test, then the KataGo final-position features -> features/final_result/test
# (all three used by predict_test.py by default). One worker per pass per GPU
# (4 KataGo in total); set PER_GPU=1 to run one KataGo per GPU when other jobs leave little GPU memory (each needs ~1 GB).
# Resumable: finished shards are skipped, and a worker whose KataGo dies is restarted.
# Overrides (defaults = the test-day layout): SHARDS, PLAYER_OUT, OPP_OUT, FINAL_OUT, LOG_DIR (logs + run tag prefix),
# GPUS (space-separated CUDA indices, default "0 1"). With PER_GPU=1 and a single GPU the two passes run one after the other.
# ATTEMPT_TIMEOUT (seconds, default 5400) bounds each extraction attempt, so a stalled engine is restarted instead of hanging.
# DETERMINISTIC=1 fixes KataGo's board symmetry (nnRandomize = false): repeated runs then give the same features (default: a
# random symmetry per evaluation, as in all training/validation features).
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SHARDS=${SHARDS:-features/shards/test}
PLAYER_OUT=${PLAYER_OUT:-features/feat96/test}
OPP_OUT=${OPP_OUT:-features/feat96opp/test}
FINAL_OUT=${FINAL_OUT:-features/final_result/test}
LOG_DIR=${LOG_DIR:-logs}
read -r -a GPU_LIST <<< "${GPUS:-0 1}"
N=${#GPU_LIST[@]}
DET=""; [ "${DETERMINISTIC:-0}" = 1 ] && DET="--deterministic"
A="--samples-per-game 96 --profiles rank_5k,rank_1d,rank_5d --depth 3 $DET"
mkdir -p "$LOG_DIR"
run() {  # name output extra_flags device worker num_workers
  for attempt in 1 2 3 4 5 6; do
    KATAGO_STDERR=$LOG_DIR/test_$1_w$5.katago.log timeout -k 60 "${ATTEMPT_TIMEOUT:-5400}" c1/.venv/bin/python extract_rank_features.py --shard-dir "$SHARDS" --output "$2" \
      $A $3 --device "$4" --worker "$5" --num-workers "$6" >> "$LOG_DIR/test_$1_w$5.log" 2>&1 && return
    echo "attempt $attempt failed or timed out, restarting in 60s" >> "$LOG_DIR/test_$1_w$5.log"; sleep 60
  done
}
if [ "${PER_GPU:-2}" = 1 ] && [ "$N" = 1 ]; then   # one GPU, one KataGo at a time
  run player "$PLAYER_OUT" "" "${GPU_LIST[0]}" 0 1
  run opponent "$OPP_OUT" --opponent "${GPU_LIST[0]}" 0 1
elif [ "${PER_GPU:-2}" = 1 ]; then   # first GPU player pass, second GPU opponent pass
  run player "$PLAYER_OUT" "" "${GPU_LIST[0]}" 0 1 &
  run opponent "$OPP_OUT" --opponent "${GPU_LIST[1]}" 0 1 &
else
  for w in "${!GPU_LIST[@]}"; do
    run player "$PLAYER_OUT" "" "${GPU_LIST[$w]}" "$w" "$N" &
    run opponent "$OPP_OUT" --opponent "${GPU_LIST[$w]}" "$w" "$N" &
  done
fi
wait
# KataGo final-position result proxy for the shift residual (fast: one 64-visit query per game)
for w in "${!GPU_LIST[@]}"; do
  timeout -k 60 "${ATTEMPT_TIMEOUT:-5400}" c1/.venv/bin/python scripts/extract_final_result.py --shard-dir "$SHARDS" --output "$FINAL_OUT" --device "${GPU_LIST[$w]}" \
    --worker "$w" --num-workers "$N" $DET >> "$LOG_DIR/test_final_w$w.log" 2>&1 &
done
wait
echo "final-result shards: $(ls "$FINAL_OUT"/feat_*.npz 2>/dev/null | wc -l)"
echo "player shards: $(ls "$PLAYER_OUT"/feat_*.npz 2>/dev/null | wc -l), opponent shards: $(ls "$OPP_OUT"/feat_*.npz 2>/dev/null | wc -l), input shards: $(ls "$SHARDS"/games_*.csv | wc -l)"
