# Adaptive uncertainty / dynamic temperature pilot

Fixed before candidate fitting/evaluation, 2026-11-07. Continues issue #1.
The 2,000-player cross-fit cohort has been repeatedly studied; every result
here is exploratory and cannot justify production adoption by itself.
No official test data or Public scores enter fitting or model selection.

## P0 audit

Use cached per-game d96, ResNet and opponent probabilities. Audit per-game
expected-rank variation, centered log-probability disagreement, member and
own/opponent disagreement, game count, frozen baseline entropy and expected-
utility top-two margin. Measure association with realized baseline utility
and leave-one-game-out decision instability, separately in each upstream half.

LOO and repeated subset diagnostics use a clearly labeled UNWEIGHTED
geometric-ensemble pooling proxy; they do not rerun the fitted nonlinear C1
correction and are not reported as full C1+C2 subset scores. Keep official
full-question C1+C2 probabilities unchanged for the calibration experiment.
For subset sizes 5/8/12/16/20, use 20 seeded samples without replacement per
player where enough games exist. Compare on the SAME players having 20 games
as well as all eligible players; report eligibility counts. Aggregate repeats
within player and never treat repeated subsets as independent observations.

One fixed 9-point temperature grid [0.5,2] is a diagnostic oracle bound on
reachable decision changes. The true-label oracle is NOT fitted, deployed,
or used to choose the uncertainty scalar, bounds or parameters.

## Single primary uncertainty scalar and model

Let s_i be the expected rank of the arithmetic mean of the three cached
member probability vectors for game i. Lock u=population_std(s_i)/sqrt(N).
This is a finite-game uncertainty proxy, not a claimed calibrated standard
error under independent games. No feature search following the P0 audit.

Fit train-only standardization of u within each fold. Dynamic model:
T=exp(clip(a+b*u_standardized, log(0.5),log(2))). Exactly two learned
parameters, initialized to zero; do not retrain upstream encoders. Use
softmax(log(max(p_base,1e-9))/T), CE + 0.01*b^2, Adam lr 0.02, 200 full-batch
steps, seed 20261107. CE is the only candidate training loss in this pilot;
no utility-surrogate search or early stopping.

Controls: unchanged identity C1+C2, numerical T=1 parity, and one global
temperature with the same bounds, 200-step CE optimizer and train-fold
fitting. The global control must reproduce the previous ~0.71695994 utility.
Use the saved five stratified player folds WITHIN each upstream-excluded
half. Never mix upstream halves when fitting calibrators. Check player,
label and half alignment; each player has one question.

## Metrics and decision rule

Preserve expected-utility decoding (exact=1, adjacent=exp(-1), otherwise=0).
Report utility, exact, large-error rate, NLL/Brier/ECE, boundary AUC,
rank/count/half subgroups, decision-change wins/losses, and per-question
temperature distributions and fitted parameters. Use 5,000 paired player
bootstrap resamples vs BOTH identity and global temperature. Secondary
subgroups are descriptive and not independently significant discoveries.

Exploratory-interest gate: >=+0.003 utility vs identity, positive 95% lower
bound vs identity AND global-T, positive gains in both halves, no >0.005
increase in >=2-rank error and no true-rank utility drop below -0.03.
If it fails, stop and do not schedule a new cohort or ordinal scale head.
If it passes, lock this one recipe before planning an untouched outer cohort
with upstream models excluding its players; that confirmation is required
before any optional production integration. A calibration-only gain is a
no-go. No changes to production predictions or submission generation.
