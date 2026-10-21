# Why are A and 1D hard to separate? (2026-10-20)

Validation-cohort A/1D adjacent AUC of the production strength score is only 0.61 (1D/2D 0.72, 4D/5D 0.73).
Three checks, all on the same 8,000 simulated validation questions as `experiments/strength_axis` (5-20 games each).

Scripts: `run.py` (builds per-question features, cached in `results/question_features.csv`; Exp 3 proxies), `train_fit.py`
(the reliable Exp 1/2 evaluation). Results: `results/report.json`, `results/train_fit.json`, logs `results/*.log`.

## Evaluation caveat (first version was wrong)
`run.py` originally fitted the extra features with 5-fold CV over only the 80 validation players of a boundary. That gave
AUCs far below 0.5 (exp1 0.27, still 0.23 after stratifying folds): with weak features and few players, leaving players out moves
the training class means away from them, so the CV is strongly biased downward. Those CV numbers in `report.json` are NOT used.
`train_fit.py` fits the feature combiners on the training cohort (2 games/player; 1,542 A, 393 1D players), freezes them and scores
validation once; only the blend s + w*z is fitted on validation (2 coefficients, stratified player-disjoint CV, player bootstrap).

## Exp 1 - relative probability under adjacent HumanSL ranks (rank_2k vs rank_1d, 32 decisions, features/feat)
| boundary | s | frozen z (val) | s + z | diff vs s [95%] | corr(s, z) |
| --- | --- | --- | --- | --- | --- |
| A/1D | 0.609 | 0.545 | 0.624 | +0.024 [-0.057, +0.111] | 0.80 |
| 1D/2D | 0.716 | 0.714 | 0.700 | -0.004 [-0.032, +0.021] | 0.83 |
| 4D/5D | 0.729 | 0.680 | 0.707 | -0.012 [-0.027, +0.001] | 0.79 |

Even on the training cohort the A/1D model reaches only 0.605 (train CV). The 1d-vs-2k likelihood ratio is mostly another view of
the same strength axis (corr 0.8) and does not separate A from 1D. Not worth a dense re-extraction of more HumanSL profiles.

## Exp 2 - error profile (surprisal quantiles / extreme shares / phases; score and win-rate loss)
No set improves on s at any boundary (A/1D: error profile -0.058 [-0.090, -0.032], surprisal-only -0.014 [-0.090, +0.060]).
Score-loss features flip direction between train and validation (A/1D train-CV 0.546, validation 0.446) - no stable signal.
Single-feature curiosity: endgame surprisal alone gives 0.65 (lower = 1D) at A/1D on validation, but it does not survive the
train-fitted combination either.

## Exp 3 - label reliability (CSV/SGF have NO ratings, dates, results, komi or promotion records: proxies only)
| rank | players | median games/player | players < 20 games |
| --- | --- | --- | --- |
| A | 1,582 | 24 | 44% |
| 1D | 433 | 87 | 1% |
| (B / 2D for reference) | 1,048 / 610 | 37 / 75 | 33% / 6% |

A is the largest, thinnest class (many players with few games); 1D the smallest with the most games per player. Within-rank
spread (IQR) of the question strength is largest for A (0.46) and 4D (0.45) vs 0.36-0.39 elsewhere. Together with Exp 1/2 this
points at the A/1D boundary being a label-definition / pool-composition problem rather than missing evidence in the moves, but it
cannot be proven without metadata.

Opponent-strength proxy: the first run reported corr(player, opponent) = 0.992 - that was a BUG: `extract_rank_features.py
--opponent` ignored the flag and re-extracted the player's own moves (see `experiments/opponent_side/README.md`). Fixed
2026-10-20; the opponent part of `run.py` is re-run once the corrected features exist.

## Update 2026-10-20 evening: opponent evidence (corrected features)
Expected rank (0 = D ... 9 = 6D) of the d96_h model, validation players, from their own moves / from their opponents' moves:
A 3.63 / 3.52, 1D 3.82 / 3.81 - a gap of only ~0.2-0.3 rank units, versus ~1.1 for B->A, ~0.7 for 1D->2D and ~0.8 for 2D->3D.
Opponents are matched by the server rank, yet the opponents of A and 1D players are almost equally strong, and the opponent
evidence separates A/1D no better (question AUC 0.64; player 0.61, both 0.63) while it lifts most other boundaries to 0.82-0.92.
So the two labels describe nearly the same playing strength in this data; it is a property of the labels/pools, not a missing
feature. A is also the most heterogeneous class (sd of own expected rank 0.92, the largest; 44% of A players have < 20 games),
consistent with a mix of genuine A players and stronger, under-ranked accounts - a plausible but unprovable explanation without
ratings or dates.
