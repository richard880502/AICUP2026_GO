#!/usr/bin/env bash
# Runs the pinned KataGo binary with CUDA libraries supplied by this venv.

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_libraries=(
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cublas/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cusolver/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/cusparse/lib"
  "$project_root/.venv/lib/python3.11/site-packages/nvidia/curand/lib"
)

export LD_LIBRARY_PATH="$(IFS=:; echo "${cuda_libraries[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$project_root/third_party/katago/katago" "$@"
