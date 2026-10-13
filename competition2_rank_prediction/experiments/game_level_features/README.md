# Game-level features (2026-10-12) - rejected

`transformer_rank.py --game-feats` appends log total moves, own pass count and ends-with-pass to every decision token (dense data, full size, single seed):
per-game top-1 0.3090 vs 0.3056 baseline, best NLL 1.691 vs 1.688 - no gain.
