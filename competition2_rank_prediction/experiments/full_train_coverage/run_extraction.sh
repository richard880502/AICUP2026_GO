#!/usr/bin/env bash
# Extracts the fixed full-player coverage sample on one selected GPU.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
research_root="$repo_root/runtime/competition2_research"
run_root="$repo_root/production/full_train_coverage_20261109"
input_shards="$run_root/shards"
feature_dir="$run_root/features"
log_dir="$run_root/logs"
workers=${NUM_WORKERS:-6}
device=${KATAGO_DEVICE:-0}
python="$research_root/c1/.venv/bin/python"

if [[ ! -x "$python" ]]; then
  echo "Research Python environment not found: $python" >&2
  exit 1
fi
if [[ ! -d "$input_shards" ]]; then
  echo "Prepared input shards not found: $input_shards" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
mkdir -p "$feature_dir" "$log_dir"
pids=()
for ((worker = 0; worker < workers; worker++)); do
  "$python" "$research_root/extract_rank_features.py" \
    --shard-dir "$input_shards" \
    --output "$feature_dir" \
    --samples-per-game 96 \
    --profiles rank_5k,rank_1d,rank_5d \
    --device "$device" \
    --worker "$worker" \
    --num-workers "$workers" \
    --depth 3 \
    > "$log_dir/worker_${worker}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if ((failed)); then
  echo "At least one extraction worker failed; completed shards can be resumed." >&2
  exit 1
fi
echo "All extraction workers completed."
