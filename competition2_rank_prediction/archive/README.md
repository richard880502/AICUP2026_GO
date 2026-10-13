# Archived rank-prediction artifacts

This directory contains recoverable historical artifacts that are not needed
by the current production inference path.

- `unused_epochs/`: ResNet epochs outside the snapshot ranges used by
  `predict_test.py`.
- `ablations/results/`: old profile-count, token-stride, learning-curve,
  legacy Transformer, and deprecated ResNet experiments.
- `ablations/logs/`: corresponding training logs.

Nothing here was deleted. Files can be moved back to `results/` if an old
experiment must be reproduced exactly.
