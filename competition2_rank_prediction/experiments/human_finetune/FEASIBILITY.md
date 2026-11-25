# Last-two-block HumanSL adaptation: feasible, no rank result yet

2026-11-09, isolated GPU1 probe on four original **training** games,
eight own decisions each. Neutral fixed1d metadata for every sample; no
true-rank input metadata. Synthetic four-class labels were used solely to
test gradients. All updated weights were discarded, not checkpointed.

- Official raw HumanSL checkpoint and pinned upstream model were loaded
  through the existing verified bridge. Vendor and extraction sources unchanged.
- Freeze everything except blocks16/17 and final trunk normalization parameters;
  keep model.eval() so normalization running statistics stay frozen.
- Trainable backbone parameters:2,954,880. Global and played-point features
  are averaged within each eight-position game, then a linear10-class head.
- Three32-position gradient updates:0.47546seconds total, including initial
  update overhead. Peak torch allocated memory705,735,168bytes on explicit
  cuda:1. This excludes CUDA context and is not a full-training estimate.
- Last-block parameters changed with finite gradients; initial spatial
  convolution stayed bit-identical. No rank utility was measured.

## Important adapter distinction

Upstream `ExtraOutputs.report()` deliberately saves `value.detach()` for
diagnostics. That is correct for the previous frozen-feature extraction, but
cannot be used unchanged for backbone fine-tuning. `GradientOutputs` is a
separate duck-typed diagnostic collector retaining only `trunkfinal` without
detaching. It does not monkey-patch or modify the official upstream files.
Two unit tests verify graph retention and independent diagnostic state.

Pylint10/10 and mypy pass for the probe. The actual GPU smoke verifies that
gradient updates are possible; declining synthetic loss is not evidence of
rank-prediction improvement. Runtime output:
`production/human_finetune_probe_20261109/report.json`.

Next required before a scored experiment: fixed player-disjoint frozen versus
adapted control, identical head/pooling/positions/sampling/update budget,
training-only input cache and preprocessing, inner-only checkpoint selection,
and complete C1+C2 conditional comparison if the pilot passes. No production
replacement or rank-training run has been launched.
