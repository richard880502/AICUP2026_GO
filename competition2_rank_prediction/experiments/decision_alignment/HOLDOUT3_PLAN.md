# Fixed two-half holdout3 confirmation

The soft decision-aligned C1 objective was fixed on the 400-player internal
calibration cohort before evaluating holdout3. This historical cohort has
been used for earlier C1/C2 research, so this is a method-specific external
confirmation, not an untouched benchmark or a Public estimate.

## Data and leakage controls

- Use the existing2,000-player holdout3 cross-fitted predictions and labels.
- For each half, use only models whose training exclusion list exactly
  matches the scored half's players.
- Fit fusion weights and C1 residuals on the separate 400-player validation
  cohort; no holdout3 labels enter fitting, normalization or selection.
- Use the current C1+C2 members d96, ResNet8c and `opp_scratch`. Keep all game
  predictions and final-position features fixed across calibration arms.
- Compare NLL C1 against the fixed soft decision-aligned C1 objective:
  temperature0.1, utility coefficient0.2, 300 steps and all other residual
  settings unchanged.

## Decision rule

Report the paired-player95% interval and results for both halves. Consider the
method externally supported only if aggregate utility improves by at least
0.003, the interval lower bound exceeds0, and both halves improve. This
historical screen cannot by itself authorize replacing the preserved Public
submission.
