# Bounded score-aware original-member fusion: proposed fixed screen

Written before implementing/fitting this specific fusion objective. Motivation
is the shared-error audit, not a positive new score. Preserve original C1+C2.

Use the original400-player validation cache only, five rank-stratified player
folds with seed20261108, original20questions/player sampling. All upstream
neural members remain frozen and exclude these players. No official test data.

For each outer fold, fit original geometric weights and C1 only on the other
320players. Freeze C1 and its feature scaler. Let baseline weights be w0.
Fit three raw parameters a (two effective relative degrees of freedom):

`w = sum(w0) * normalize(w0 * exp(0.2*tanh(a)))`.

This retains nonnegativity and total scale; no global temperature or class
bias is learned. Initial a=0 must reproduce original corrected predictions
and decisions. C1 features use original own/opponent/final inputs and thus
remain fixed as member weights vary. Mean per-member game log-probabilities
provide differentiable question logits; class-independent normalization
constants can be dropped before the final softmax.

Matched arms:

- Original: unchanged per-game-NLL weights and original C1.
- NLL control: bounded relative weights fit to question NLL plus
  `0.1*mean((0.2*tanh(a))**2)`.
- Utility: same parameters/regularizer, maximize smooth decision reward.
  For posterior p and official reward matrix R, use decision distribution
  `softmax((p @ R)/0.1)` and loss negative expected R[true_label,decision].

Both new arms: initialize a=0, Adamlr0.02,300full-batch steps, no checkpoint
or hyperparameter selection. All labels used for fitting stay in the320-player
outer-training fold. No outer probabilities/labels tune weights. The baseline
C1 may use those same fitting labels as ordinary supervised training; it must
never use outer80player labels. This is not a router trained on other folds'
globally precomputed OOF calibration outputs.

Primary contrasts utility−original and utility−NLL, paired-player97.5% bootstrap
intervals. Report per-rank/query-size changes and decision gain/loss counts.
Screen requires utility−original>=0.005, both interval lower bounds>0,
and no rank loss>0.05. No grid expansion if the fixed screen fails. Historical
cohort reuse makes this exploratory even if it passes; separate confirmation
would still be needed.

Implementation note before complete scoring: an initial simplified
mean-member-log representation failed baseline-posterior parity on fold4,
because production clips fused per-game probabilities at1e-6before pooling.
That attempt stopped without a completed report. The corrected implementation
retains the exact per-game log-softmax, floor and masked question averaging
inside autograd. Objective, bounds, steps, folds and acceptance thresholds
are unchanged; parity tolerance was not weakened. A regression test covers
extreme probabilities and unequal/padded query lengths.
