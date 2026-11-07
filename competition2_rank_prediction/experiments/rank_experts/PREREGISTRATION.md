# Frozen rank experts: exploratory pilot

Recorded before fitting or evaluating any expert, 2026-11-06.

Use the existing 2,000-player C1+C2 OOF caches. This cohort has been studied
before: results are exploratory and cannot confirm production adoption.
No official test labels, predictions, or leaderboard scores enter fitting.

Keep upstream models frozen. Each cache half uses models excluding that
entire half. Fit experts using stratified five-fold splits WITHIN each half;
never train an expert on the opposite half, whose upstream models may have
seen its evaluation players. Fit feature scalers on expert training rows only.
Audit unique underlying players, half exclusion lists and question game counts.
Zero-initialized corrections must reproduce cached probabilities and decisions.

Fixed candidates: bias-only; linear bounded residual; three overlapping region
MLPs with fixed normalized probability-mass gates; three boundary MLPs with
frozen pair-mass gates and pair-mass-preserving corrections. Regions are
D/C/B/A, B/A/1D/2D/3D, and 2D/3D/4D/5D/6D. Boundaries are A/1D, 1D/2D,
4D/5D. Inputs: centered base logits, per-member question probability moments,
own/opponent strength-gap moments, final-position moments and log game count.
MLP width 32, residual bound 0.5, 200 full-batch AdamW steps, learning rate
0.003, weight decay 0.01, correction penalty 0.01. Seed 20261106.
No early stopping, hyperparameter search, learned gate or combined model.

Primary candidate: region MLP vs frozen baseline AND bias-only / linear
controls. Utility = exact 1, adjacent exp(-1), otherwise 0; retain expected
utility decoding. Report exact accuracy, large errors, rank, game-count,
boundary AUC, and paired player bootstrap intervals. One question per player.
Bootstrap 5,000 replicates; secondary comparisons are descriptive, not claims
of significance. A promising candidate needs >= +0.005 utility, positive
95% lower paired bound vs baseline and controls, and no increase >0.005 in
two-or-more-rank errors, before considering a fresh-cohort confirmation.
No experiment results may replace production automatically.
