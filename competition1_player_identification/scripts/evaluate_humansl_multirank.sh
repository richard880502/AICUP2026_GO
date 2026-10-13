#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin="${project_root}/.venv/bin/python"
base_cache="${project_root}/features/best/spatial_ownership_s64"
human_cache="${project_root}/features/experiments/humansl_rank1d_rank5d_s64_full"
experiment_root="${project_root}/features/experiments/humansl_multirank_merged"
profile_cache="${experiment_root}/profiles.npz"
result_path="${project_root}/results/experiments/humansl_multirank_arcface.json"

if [[ ! -f "${human_cache}/candidates.npz" || \
      ! -f "${human_cache}/queries.npz" ]]; then
  echo "HumanSL extraction is not complete: ${human_cache}" >&2
  exit 1
fi

if [[ -e "${experiment_root}" || -e "${result_path}" ]]; then
  echo "Refusing to overwrite HumanSL experiment artifacts" >&2
  exit 1
fi

mkdir -p "${project_root}/features/experiments"
mkdir -p "${project_root}/results/experiments"

"${python_bin}" "${project_root}/merge_decision_feature_caches.py" \
  --base "${base_cache}" \
  --supplement "${human_cache}" \
  --feature-prefix human_rank_ \
  --output "${experiment_root}"

"${python_bin}" "${project_root}/cache_rich_profiles.py" \
  --features "${experiment_root}" \
  --output "${profile_cache}" \
  --workers 12

"${python_bin}" \
  "${project_root}/evaluate_arcface_cross_dimension_ensemble.py" \
  --profiles "${profile_cache}" \
  --output "${result_path}" \
  --blend-by-query-count \
  --device cuda:0
