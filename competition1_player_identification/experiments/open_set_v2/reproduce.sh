#!/usr/bin/env bash
# Replays a sealed experiment, or reproduces all stages from authorized inputs.
set -euo pipefail

project_root=$(git rev-parse --show-toplevel)
cd "$project_root"
python_bin="$project_root/runtime/competition1_research/.venv/bin/python"
experiment_root=production/open_set_v2_20261111
package=competition1_player_identification.experiments.open_set_v2
export PYTHONPATH="$project_root:$project_root/competition1_player_identification"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLBACKEND=Agg

run_stage() {
  local module=$1
  shift
  "$python_bin" -u -m "$package.$module" "$@"
}

if [[ -f "$experiment_root/locked_configuration.json" ]]; then
  run_stage final_evaluate
  run_stage findings
  exit 0
fi

run_stage prepare
[[ -f "$experiment_root/opening_validation.json" ]] || run_stage opening
[[ -f "$experiment_root/prefix_validation.json" ]] || run_stage prefix
for kind in opening sequence; do
  if [[ ! -f "$experiment_root/encoder_$kind/completed.json" ]]; then
    run_stage train --kind "$kind" --device cuda:0
  fi
done
[[ -f "$experiment_root/rank_prior/completed.json" ]] || run_stage rank_prior

if [[ ! -f "$experiment_root/engine_profile_audit.json" ]]; then
  extractor_args=(
    --candidates "$experiment_root/engine_inputs/games.csv"
    --questions "$experiment_root/engine_inputs/questions.csv"
    --output "$experiment_root/engine_features"
    --engine runtime/competition1_research/scripts/run_katago.sh
    --engine-config competition1_player_identification/configs/analysis_1x3090.cfg
    --model runtime/competition1_research/models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
    --human-model runtime/competition1_research/models/katago/b18c384nbt-humanv0.bin.gz
    --engine-batch-size 16 --shard-size 128 --num-workers 2 --dynamic-shards
  )
  CUDA_VISIBLE_DEVICES=0 "$python_bin" -u -m \
    competition1_player_identification.extract_submission_features \
    "${extractor_args[@]}" --worker-index 0 &
  worker_zero=$!
  CUDA_VISIBLE_DEVICES=1 "$python_bin" -u -m \
    competition1_player_identification.extract_submission_features \
    "${extractor_args[@]}" --worker-index 1 &
  worker_one=$!
  cleanup_workers() {
    kill -INT "$worker_zero" "$worker_one" 2>/dev/null || true
  }
  trap cleanup_workers EXIT
  wait "$worker_zero"
  wait "$worker_one"
  trap - EXIT
  run_stage engine_profiles --workers 8
fi
[[ -f "$experiment_root/fisher_parameters.npz" ]] || run_stage profile_baseline
for dimension in 512 1024; do
  if [[ ! -f "$experiment_root/encoder_engine_$dimension/completed.json" ]]; then
    run_stage train --kind engine --dimension "$dimension" --epochs 32 \
      --device cuda:0 --output "$experiment_root/encoder_engine_$dimension"
  fi
done
[[ -f "$experiment_root/rank_engine/completed.json" ]] || run_stage decision_rank
[[ -f "$experiment_root/matching/completed.json" ]] || run_stage matching
run_stage fusion
run_stage replay
run_stage seal
run_stage final_evaluate
run_stage findings
