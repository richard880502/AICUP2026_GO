#!/usr/bin/env bash
# Reproduces P2 end to end from the research root:
#   1. train tenp3_c{0,1} / tenp10_c{0,1}            (p2_train.sh)
#   2. require a complete ten-profile holdout3 cache  (check_cache.py)
#   3. score both folds, verifying val_probs_*.npz    (p2_score.py)
#   4. stage A (C1+C2) and stage B (v47) replays      (p2_evaluate.py)
# The holdout3 cache is extracted like the training cache (run_p1.sh's
# extract_set or lowmem_p1.sh) before step 2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY=${PY:-c1/.venv/bin/python}
OUT=${OUT:-/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production}
P2=${P2:-$OUT/p2}
VAL_FEATURES=${VAL_FEATURES:-/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit/feat10_val}
H3_FEATURES=${H3_FEATURES:-$OUT/feat10_holdout3}
GPU=${GPU:-1}
MIN_AVAILABLE_KB=${MIN_AVAILABLE_KB:-5242880}
export OUT P2 VAL_FEATURES MIN_AVAILABLE_KB
export TRAIN_GPU="$GPU"

if [[ "$GPU" == "0" ]]; then
  echo "GPU 0 hosts another service; refusing to use it" >&2
  exit 1
fi

wait_for_memory() {
  local available
  while true; do
    available=$(awk '/^MemAvailable/{print $2}' /proc/meminfo)
    (( available >= MIN_AVAILABLE_KB )) && return
    echo "$(date '+%F %T') MemAvailable ${available} kB < ${MIN_AVAILABLE_KB} kB; waiting"
    sleep 60
  done
}

bash experiments/ten_profile_production/p2_train.sh

"$PY" -m experiments.ten_profile_production.check_cache \
  --features "$H3_FEATURES" --shard-dir features/shards/holdout3 \
  | tee "$P2/logs/check_cache_holdout3.log"

for fold in 0 1; do
  if [[ -f "$P2/probabilities/score_c${fold}.json" ]]; then
    echo "[score] reusing fold $fold"
    continue
  fi
  wait_for_memory
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.ten_profile_production.p2_score \
    --fold "$fold" \
    --model-dir "$P2/models" \
    --validation "$VAL_FEATURES" \
    --holdout3 "$H3_FEATURES" \
    --output-dir "$P2/probabilities" \
    --device cuda:0 \
    2>&1 | tee "$P2/logs/score_c${fold}.log"
done

wait_for_memory
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.ten_profile_production.p2_evaluate \
  --probability-dir "$P2/probabilities" \
  --output "$P2/p2_report.json" \
  --device cuda:0 \
  2>&1 | tee "$P2/logs/evaluate_p2.log"

echo "P2 report: $P2/p2_report.json"
