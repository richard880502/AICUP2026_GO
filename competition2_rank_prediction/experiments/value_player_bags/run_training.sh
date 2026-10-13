#!/usr/bin/env bash
# Builds value-augmented embeddings, then trains and validates one bag branch.
# This script deliberately never reads the internal holdout split.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

device=${DEVICE:-cuda:0}
embeddings=experiments/value_player_bags/embeddings_logits_moves_value
results=experiments/value_player_bags/results/transformer_full

c1/.venv/bin/python -m experiments.value_player_bags.attach_value_statistics \
  --source experiments/dense_player_bags/embeddings_logits_moves \
  --output "$embeddings" \
  --splits train validation

c1/.venv/bin/python -m experiments.dense_player_bags.train_bags \
  --embeddings "$embeddings" \
  --output "$results" \
  --architecture transformer \
  --device "$device"

c1/.venv/bin/python -m experiments.dense_player_bags.evaluate_validation \
  --embeddings "$embeddings" \
  --checkpoint "$results/model.pt" \
  --device "$device" \
  --output "$results/validation_fusion.json"
