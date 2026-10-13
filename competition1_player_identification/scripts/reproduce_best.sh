#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

  "${project_root}/.venv/bin/python" \
  "${project_root}/evaluate_arcface_cross_dimension_ensemble.py" \
  --profiles "${project_root}/features/best/humansl_multirank_s64/profiles.npz" \
  --output "${project_root}/results/best/reproduced_humansl_multirank_arcface.json" \
  --blend-by-query-count \
  --device cuda:0
