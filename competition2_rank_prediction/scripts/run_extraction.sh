#!/usr/bin/env bash
# Resumable extraction with NUM_WORKERS KataGo processes sharing one GPU (default: 2 on GPU 1).
# Validation cohort first, then train shards. Safe to re-run: finished shards are skipped.
#   NUM_WORKERS=2 DEVICE=1 scripts/run_extraction.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
NUM_WORKERS=${NUM_WORKERS:-2}
DEVICE=${DEVICE:-1}
for ((w = 0; w < NUM_WORKERS; w++)); do
  (
    for split in val train; do
      c1/.venv/bin/python extract_rank_features.py --shard-dir "features/shards/$split" \
        --output "features/feat/$split" --samples-per-game 32 \
        --device "$DEVICE" --worker "$w" --num-workers "$NUM_WORKERS" --depth 3
    done
  ) > "logs/extraction_w${w}.log" 2>&1 &
done
wait
