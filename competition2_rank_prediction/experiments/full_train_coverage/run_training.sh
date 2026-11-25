#!/usr/bin/env bash
# Trains two player-excluded folds of the expanded dense rank encoder.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
cd "$repo_root"
research_root="$repo_root/runtime/competition2_research"
run_root="$repo_root/production/full_train_coverage_20261109"
extra_features="$run_root/features"
base_features="$research_root/features/feat96/train"
validation_features="$research_root/features/feat96/val"
python="$research_root/c1/.venv/bin/python"
workers=${OMP_NUM_THREADS:-4}
device=${DEVICE:-cuda:0}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export OMP_NUM_THREADS="$workers"
export MKL_NUM_THREADS="$workers"
export OPENBLAS_NUM_THREADS="$workers"

expected=$(find "$run_root/shards" -maxdepth 1 -name 'games_*.csv' | wc -l)
actual=$(find "$extra_features" -maxdepth 1 -name 'feat_*.npz' | wc -l)
if [[ "$actual" -ne "$expected" ]]; then
  echo "Expected $expected extracted shards, found $actual" >&2
  exit 1
fi

for fold in 0 1; do
  train_dir="$run_root/merged_train_c${fold}"
  mkdir -p "$train_dir"
  for feature in "$base_features"/feat_*.npz "$extra_features"/feat_*.npz; do
    target="$train_dir/$(basename "$feature")"
    if [[ -e "$target" && ! -L "$target" ]]; then
      echo "Refusing to replace non-symlink feature: $target" >&2
      exit 1
    fi
    ln -sfn "$feature" "$target"
  done

  "$python" "$research_root/transformer_rank.py" \
    --train "$train_dir" \
    --val "$validation_features" \
    --exclude-players "$research_root/features/holdout3/exclude_c${fold}.txt" \
    --seeds 3 \
    --epochs 40 \
    --device "$device" \
    --out "$research_root/results" \
    --tag "d96_fullcov_c${fold}"

  PYTHONPATH="$research_root${PYTHONPATH:+:$PYTHONPATH}" \
    "$python" -m \
    competition2_rank_prediction.experiments.full_train_coverage.score \
    --fold "$fold" \
    --device "$device"
done

PYTHONPATH="$research_root${PYTHONPATH:+:$PYTHONPATH}" \
  "$python" -m \
  competition2_rank_prediction.experiments.full_train_coverage.evaluate
