# Targeted final-winrate volatility follow-up

Fixed before evaluating the targeted models, 2026-11-07. This follows an
exploratory maximum correlation among 79 features on an already-studied
2,000-player cohort, so neither p-values nor intervals establish a new
independent discovery. No official test data or scores are used.

Hypothesis: within-question std(final_winrate) contains a small rank-error
signal beyond C1+C2. Distinguish it from nonlinear outcome balance and game
count: var(w) = mean(w)*(1-mean(w)) - mean(w*(1-w)). Terminal winrates often
approach zero or one, making volatility largely an outcome-balance statistic.

Use the same saved, within-half five-fold player splits. The primary feature
is volatility residualized against centered baseline logits, mean winrate,
mean-winrate squared and log game count, with train-only standardized ridge
(lambda 10). Fit a one-feature scalar rank-axis correction plus scalar bias,
bounded by 0.5*tanh(b+w*x), with axis linspace(-1.5,1.5,10), CE + 0.1*w^2,
300 Adam steps at lr 0.02, seed 20261107. Fit all normalization on training
rows. Controls: frozen baseline and the same axis-bias-only model. Raw
volatility correction is a secondary explanatory control, not another
production candidate. No search or early stopping.

Report raw and conditional correlations in each upstream half; nuisance
reconstruction R2; primary coefficients; paired player-bootstrap utility,
NLL and Brier differences (5,000 replicates); rank/count subgroups; and
decision wins/losses. Reproduce the original baseline utility exactly.

Pilot-interest gate: primary >= +0.003 utility, paired 95% lower bound > 0
against BOTH frozen baseline and bias-only; positive utility deltas in both
halves; no >0.005 rise in >=2-rank errors. Even passing only justifies a new
independent confirmation with a fixed recipe, not production adoption.
If it fails, report whether a weak statistical association persists without
useful retrieval/classification gain; do not search until a positive appears.
