#!/usr/bin/env bash
# Waits for existing training jobs, then scores the complete frozen candidate.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
cd "$repo_root"
research_root="$repo_root/runtime/competition2_research"
run_root="$repo_root/production/dense_encoder_scale_20261109"
checkpoints="$research_root/experiments/dense_encoder_scale/results"
python="$research_root/c1/.venv/bin/python"
mkdir -p "$run_root/logs"
exec > >(tee -a "$run_root/logs/postprocess.log") 2>&1

for training_pid in "$@"; do
  if [[ ! "$training_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid training PID: $training_pid" >&2
    exit 1
  fi
  while process_state=$(ps -p "$training_pid" -o stat=); do
    if [[ "$process_state" == Z* ]]; then
      break
    fi
    sleep 30
  done
done

for fold in 0 1; do
  for seed in 0 1 2; do
    checkpoint="$checkpoints/d96_densebag_c${fold}_seed${seed}.pt"
    if [[ ! -s "$checkpoint" ]]; then
      echo "Training incomplete; missing checkpoint: $checkpoint" >&2
      exit 1
    fi
  done
  if [[ ! -s "$checkpoints/val_report_d96_densebag_c${fold}.json" ]]; then
    echo "Training incomplete; fold $fold has no final validation report" >&2
    exit 1
  fi
done

export PYTHONPATH="$research_root${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
for fold in 0 1; do
  "$python" -m competition2_rank_prediction.experiments.full_train_coverage.score \
    --fold "$fold" --device cpu --tag-prefix d96_densebag \
    --checkpoint-dir "$checkpoints" \
    --probability-dir "$run_root/probabilities"
done
"$python" -m competition2_rank_prediction.experiments.full_train_coverage.evaluate \
  --candidate-name densebag \
  --probability-dir "$run_root/probabilities" \
  --output-report "$run_root/crossfit_report.json"
date -Is > "$run_root/postprocess_completed.txt"
