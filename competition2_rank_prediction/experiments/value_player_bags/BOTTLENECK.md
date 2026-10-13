# Rank-prediction bottleneck note

Date: 2026-10-07 (Taipei).

The strongest frozen cross-feature bag model was evaluated on ten independent,
player-disjoint validation cohorts of 400 players, with 40 players per rank.
Its ordinary per-question decode score is:

| Query games | Score |
| ---: | ---: |
| 5 | 0.60687 |
| 10 | 0.66205 |
| 15 | 0.67740 |
| 20 | 0.69204 |

An exploratory least-squares fit of `score = a - b / sqrt(N)` gives
`a = 0.77704` and residuals below 0.0044 at all four observed query sizes.
This is not a formal Bayes limit, but it is strong evidence that merely
averaging more instances of the current policy/style representation will not
reliably reach 0.8 at the fixed 5--20-game test sizes. Replacing the bag
Transformer or adding a class-count constraint is therefore low priority.

The class-count hypothesis was tested in the same validation protocol. A
Hungarian assignment constrained to exactly 40 questions per rank improved
only the 5-game setting (+0.00455) and reduced 10/15/20-game scores by
0.00292/0.00364/0.00974. It is rejected even under a known balanced label
distribution.

## Value-only lower bound

A shrinkage-LDA classifier trained on 15,211 earlier value-feature games and
evaluated on the disjoint 8,000-game validation cohort reached 19.3% per-game
top-1 accuracy. After log-mean pooling, its 5/10/15/20-game utility scores
were 0.3511/0.3437/0.3473/0.3293. Therefore value and ownership summaries are
not a standalone rank predictor. The active experiment treats them only as an
independent, nonlinear complementary branch to the 606D policy/style model;
it must demonstrate validation fusion lift to remain justified.

## Complete value-bag result: rejected

The full experiment added normal-KataGo win-rate loss, score-loss, and
ownership-change summaries to all 20 games in each player bag. It produced a
modest, repeatable validation complement to the production pipeline:

| Validation question sampling seed | Cross-feature | Cross-feature + value |
| ---: | ---: | ---: |
| 20261009 | 0.65688 | 0.65816 |
| 20261019 | 0.65901 | 0.66128 |

The value-branch weight (0.12) was selected on the first sampling and then
confirmed unchanged on the second. The single locked CPU holdout evaluation,
recorded in `results/transformer_full/holdout_cross_value_locked_cpu.json`,
did not transfer: it changed the cross-feature score from **0.677955** to
**0.675856** (delta -0.002098; paired 95% interval [-0.019482, 0.015544]).
The branch is therefore rejected from the best pipeline. This is evidence that
the value/ownership signal is real but cohort-sensitive and presently too weak
to close the gap to 0.8.
