# Soft decision-aligned C1 calibration screen

## Hypothesis

The previous mixed C1 objective rewards probability mass on the true and
adjacent classes. Production instead decodes a single rank by maximizing
expected official reward. These are different training targets.

This screen replaces only the C1 residual objective with a differentiable
soft-decision approximation. First compute class probabilities `p`, then
expected score for each submitted rank `q = p @ reward.T`. A temperature
`0.1` softmax over `q` approximates the decoder's selected action. Optimize
cross-entropy plus `0.2 * (1 - expected official reward of that soft action)`.

## Fixed comparison

- Same original400-player / 8,000-game cohort and five player-disjoint folds.
- Same frozen d96, ResNet8c and opponent-scratch members.
- Same per-game fusion, C1 features, sampled queries, residual architecture,
  initialization, Adam settings, regularization and 300 training steps.
- Only the C1 residual loss changes from cross-entropy.
- Fixed action temperature0.1 and utility coefficient0.2; no parameter sweep.
- Advancement requires at least+0.003 official utility and a positive paired
  player97.5% interval lower bound.
- No Public inference, submission, upload, commit or push.

This is a historical internal screen, not an independent Public estimate.
