# 1D/2D specialist pilot

Locked 2026-11-07 before extracting embeddings or evaluating the candidate.
The repeatedly explored holdout3 cohort is exploratory, not a fresh outer
cohort. No official test, Public score, or production fitting is used.

## Data and exclusion

Reuse the 2,000 holdout3 questions and saved within-half five folds. Each
question is a distinct player; each upstream half contains 100 1D and 100
2D players. For each half, use only that half's d96_c{half}_seed0 and
opp_scratch_c{half}_seed0 game encoder embeddings. Their training scripts
use exclude_c{half}.txt, which is checked against all the half's players.
Encoder game features are pooled as a per-question arithmetic mean, with
own and opponent vectors concatenated. Empty games map to zero embeddings.
This is a single 256-dimensional input, not logits. Record checkpoint hashes.
Train and evaluate specialist only within its upstream-excluded half; use
five saved folds and standardize using the specialist training fold only.
Train only players with true rank 1D/2D, evaluate all ten ranks. Each fold
should contain 160 pair players for training and 40 for held-out evaluation.

## Locked model and inference

One L2-regularized logistic regression on standardized mean embeddings,
balanced classes, C=0.01, sklearn lbfgs, max_iter=1000, seed=20261107.
No hyperparameter search or MLP in this pilot. Routing is predetermined:
baseline 1D+2D probability mass >= 0.35 AND the baseline expected-utility
decision is 1D or 2D. Routing sees no labels. For routed questions blend
baseline conditional pair log-odds and specialist log-odds equally (w=0.5),
preserve pair mass, preserve other probabilities, and decode using the
official expected utility. w=0 must reproduce baseline probabilities and
decisions exactly. Baseline utility must reproduce 0.7170117549099193.

## Metrics and decision

Report pooled and half-specific held-out pair AUC/balanced accuracy/Brier,
training vs held-out pair AUC, baseline pair-odds AUC, all-ten-class utility,
exact and >=2 errors, per-rank and game-count groups, pair confusion,
non-pair false routes, changed-decision wins/losses, and 5,000 paired player
bootstrap utility intervals. The single primary metric is all-ten-class
utility delta against frozen C1+C2. A no-go results if delta <= 0, either
half's delta <= 0, bootstrap 95% lower bound <= 0, pair AUC is not above
baseline in both halves, >=2-error increases by >0.005, or any rank loses
>0.03 utility. Positive pilot outcomes merely justify a frozen-recipe
untouched outer evaluation; no production adoption without it. Negative
results stop this route without expanding to the remaining boundaries.
