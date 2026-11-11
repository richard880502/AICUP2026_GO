#!/usr/bin/env bash
# Continues the four already running extraction jobs through final submission.
set -euo pipefail
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
cd "$repo_root"
if [[ "$#" -ne 4 ]]; then
  echo "Usage: finish.sh PUB2_PID PRIV2_PID PUB_PID PRIV_PID" >&2
  exit 1
fi
pool_names=(pub2 priv2 pub priv)
extraction_pids=("$@")
run_root="$repo_root/production/player_v2_20261111"
data_root="$repo_root/data/aicup2026_player_identification_v2_tests_20261108_fix_readme"
python="$repo_root/runtime/competition1_research/.venv/bin/python"
export PYTHONPATH="$repo_root:$repo_root/competition1_player_identification"
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p "$run_root/logs"
exec > >(tee -a "$run_root/logs/production.log") 2>&1
for index in 0 1 2 3; do
  pool=${pool_names[$index]}
  if [[ -s "$run_root/$pool/predictions.csv" ]]; then
    echo "Reusing completed $pool predictions; final validation checks all pools"
    continue
  fi
  IFS=',' read -r -a pool_pids <<< "${extraction_pids[$index]}"
  for extraction_pid in "${pool_pids[@]}"; do
  if [[ ! "$extraction_pid" =~ ^[0-9]+$ ]]; then
    echo "Invalid extraction PID: $extraction_pid" >&2
    exit 1
  fi
  echo "Waiting for $pool extraction PID $extraction_pid"
  while process_state=$(ps -p "$extraction_pid" -o stat=); do
    if [[ "$process_state" == Z* ]]; then
      break
    fi
    process_args=$(ps -p "$extraction_pid" -o args=) || break
    if [[ "$process_args" != *"player_v2_20261111/$pool/features"* ]]; then
      break
    fi
    sleep 30
  done
  done
  echo "Processing pool $pool"
  "$python" -u -m competition1_player_identification.production_pipeline.run \
    --pool "$pool" --data-dir "$data_root" --output "$run_root" \
    --device cuda:0 --workers 4
done
"$python" -u -m competition1_player_identification.production_pipeline.run \
  --pool aggregate --data-dir "$data_root" --output "$run_root"
date -Is > "$run_root/completed.txt"
