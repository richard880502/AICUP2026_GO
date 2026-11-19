#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY=${PY:-c1/.venv/bin/python}
OUT=${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production}
TRAIN_FEATURES=${TRAIN_FEATURES:-$OUT/feat10_train}
VAL_FEATURES=${VAL_FEATURES:-/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit/feat10_val}
MODEL_OUT=${MODEL_OUT:-$OUT/models}
LOG_DIR=${LOG_DIR:-$OUT/logs}
EXTRACT_GPU=${EXTRACT_GPU:-1}
TRAIN_GPU=${TRAIN_GPU:-1}
WORKERS=${WORKERS:-6}
PROFILES=rank_11k,rank_8k,rank_5k,rank_2k,rank_1d,rank_2d,rank_3d,rank_4d,rank_5d,rank_6d
DET=""
[[ "${DETERMINISTIC:-0}" == "1" ]] && DET="--deterministic"

mkdir -p "$OUT" "$MODEL_OUT" "$LOG_DIR"

check_cache() {
  "$PY" -m experiments.ten_profile_production.check_cache \
    --features "$1" --shard-dir "$2"
}

extract_set() {
  local name="$1" shard_dir="$2" feature_dir="$3"
  if check_cache "$feature_dir" "$shard_dir" >/dev/null 2>&1; then
    echo "[$name] reusing complete ten-profile cache: $feature_dir"
    return
  fi
  echo "[$name] extracting ten HumanSL profiles with $WORKERS workers on GPU $EXTRACT_GPU"
  mkdir -p "$feature_dir"
  for ((worker=0; worker<WORKERS; worker++)); do
    CUDA_VISIBLE_DEVICES="$EXTRACT_GPU" "$PY" extract_rank_features.py \
      --shard-dir "$shard_dir" \
      --output "$feature_dir" \
      --samples-per-game 96 \
      --profiles "$PROFILES" \
      --device 0 \
      --worker "$worker" \
      --num-workers "$WORKERS" \
      --depth 3 $DET \
      > "$LOG_DIR/extract_${name}_w${worker}.log" 2>&1 &
  done
  wait
  check_cache "$feature_dir" "$shard_dir"
}

# The existing rankmle validation extraction is reused when complete.
extract_set val features/shards/val "$VAL_FEATURES"
extract_set train features/shards/train "$TRAIN_FEATURES"

if [[ ! -f "$MODEL_OUT/val_probs_tenp3.npz" ]]; then
  echo "[train] matched 3-profile production arm"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" transformer_rank.py \
    --train "$TRAIN_FEATURES" \
    --val "$VAL_FEATURES" \
    --seeds 3 --epochs 40 --device cuda:0 \
    --tag tenp3 --out "$MODEL_OUT" \
    --keep-profiles 0,3,5,9 \
    > "$LOG_DIR/train_tenp3.log" 2>&1
else
  echo "[train] reusing $MODEL_OUT/val_probs_tenp3.npz"
fi

if [[ ! -f "$MODEL_OUT/val_probs_tenp10.npz" ]]; then
  echo "[train] full 10-profile production arm"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" transformer_rank.py \
    --train "$TRAIN_FEATURES" \
    --val "$VAL_FEATURES" \
    --seeds 3 --epochs 40 --device cuda:0 \
    --tag tenp10 --out "$MODEL_OUT" \
    > "$LOG_DIR/train_tenp10.log" 2>&1
else
  echo "[train] reusing $MODEL_OUT/val_probs_tenp10.npz"
fi

echo "[evaluate] matched P1 comparison"
"$PY" -m experiments.ten_profile_production.evaluate_p1 \
  --three "$MODEL_OUT/val_probs_tenp3.npz" \
  --ten "$MODEL_OUT/val_probs_tenp10.npz" \
  --output "$OUT/p1_report.json" \
  | tee "$LOG_DIR/evaluate_p1.log"

echo "P1 report: $OUT/p1_report.json"
