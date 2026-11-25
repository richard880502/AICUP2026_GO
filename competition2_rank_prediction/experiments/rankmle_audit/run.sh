#!/usr/bin/env bash
# 2026-11-18 rankmle premise audit: ten HumanSL rank profiles on the 400-player validation cohort
# (8 shards x 1,000 games = 400 players x 20 games), then the CPU-only audit. Run on w77; paths are relative to the repo
# root (runtime/ is the untracked w77-local research tree). GPU 0 hosts another service, so extraction uses GPU 1 only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

OUT=/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit
LOGS=${LOGS:-/tmp2/richard/claude_scratch}
PROFILES=rank_11k,rank_8k,rank_5k,rank_2k,rank_1d,rank_2d,rank_3d,rank_4d,rank_5d,rank_6d
mkdir -p "$OUT/feat10_val" "$LOGS"

workers=6
for ((worker = 0; worker < workers; worker++)); do
  CUDA_VISIBLE_DEVICES=1 runtime/competition2_research/c1/.venv/bin/python \
    runtime/competition2_research/extract_rank_features.py \
    --shard-dir runtime/competition2_research/features/shards/val \
    --output "$OUT/feat10_val" \
    --samples-per-game 96 \
    --profiles "$PROFILES" \
    --device 0 \
    --worker "$worker" \
    --num-workers "$workers" \
    --depth 3 \
    > "$LOGS/feat10_w${worker}.log" 2>&1 &
done
wait

PYTHONPATH=.:competition1_player_identification competition2_rank_prediction/c1/.venv/bin/python \
  -m competition2_rank_prediction.experiments.rankmle_audit.audit \
  --features "$OUT/feat10_val" \
  --output "$OUT/report.json"

# Seed robustness and the shuffled-column dimension control (CPU, ~10 minutes).
PYTHONPATH=.:competition1_player_identification competition2_rank_prediction/c1/.venv/bin/python \
  -m competition2_rank_prediction.experiments.rankmle_audit.seed_robustness \
  --features "$OUT/feat10_val" \
  --seeds 20 \
  --output "$OUT/seed_robustness.json"
