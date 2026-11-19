#!/usr/bin/env bash
# P2 stage A, step 1: trains the four holdout3 cross-fit arms.
#
#   tenp3_c{0,1}   ten-profile cache, --keep-profiles 0,3,5,9 (normal, 5k, 1d, 5d)
#   tenp10_c{0,1}  ten-profile cache, all eleven streams
#
# Everything else is the P1 recipe (transformer_rank.py defaults, 3 seeds,
# 40 epochs); fold c additionally excludes holdout3 half c's 1,000 players,
# exactly like the production d96_c{c} members. Run from the research root.
# Existing arms (val_probs_<tag>.npz present) are reused.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY=${PY:-c1/.venv/bin/python}
OUT=${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production}
P2=${P2:-$OUT/p2}
TRAIN_FEATURES=${TRAIN_FEATURES:-$OUT/feat10_train}
VAL_FEATURES=${VAL_FEATURES:-/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit/feat10_val}
TRAIN_GPU=${TRAIN_GPU:-1}
# Shared host: never start a run below this much MemAvailable (kB).
MIN_AVAILABLE_KB=${MIN_AVAILABLE_KB:-5242880}

if [[ "$TRAIN_GPU" == "0" ]]; then
  echo "GPU 0 hosts another service; refusing to train on it" >&2
  exit 1
fi

mkdir -p "$P2/models" "$P2/logs"

wait_for_memory() {
  local available
  while true; do
    available=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    if (( available >= MIN_AVAILABLE_KB )); then
      return
    fi
    echo "$(date '+%F %T') MemAvailable ${available} kB < ${MIN_AVAILABLE_KB} kB; waiting"
    sleep 60
  done
}

train_arm() {
  local arm="$1" fold="$2"
  local tag="${arm}_c${fold}"
  local keep=()
  [[ "$arm" == "tenp3" ]] && keep=(--keep-profiles 0,3,5,9)
  if [[ -f "$P2/models/val_probs_${tag}.npz" ]]; then
    echo "[train] reusing $P2/models/val_probs_${tag}.npz"
    return
  fi
  wait_for_memory
  echo "$(date '+%F %T') [train] $tag"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PY" transformer_rank.py \
    --train "$TRAIN_FEATURES" \
    --val "$VAL_FEATURES" \
    --exclude-players "features/holdout3/exclude_c${fold}.txt" \
    --seeds 3 --epochs 40 --device cuda:0 \
    --tag "$tag" --out "$P2/models" \
    "${keep[@]}" \
    > "$P2/logs/train_${tag}.log" 2>&1
  echo "$(date '+%F %T') [train] $tag done"
}

for fold in 0 1; do
  for arm in tenp3 tenp10; do
    train_arm "$arm" "$fold"
  done
done
