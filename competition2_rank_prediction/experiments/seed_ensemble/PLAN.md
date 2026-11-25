# Opponent-model seed ensemble ablation

The current C1+C2 pipeline already averages three independently initialized
`opp_scratch` checkpoints. This fixed screen tests whether any one seed
weakens the ensemble.

## Fixed conditions

- Use the existing two-half holdout3 cross-fit; each half's opponent models
  exclude exactly the scored1,000 players.
- Compare the all-three-seed mean with the three prespecified leave-one-seed
  means: seeds(1,2), (0,2), and(0,1).
- For each condition, refit the same nonnegative game-level fusion and NLL C1
  residual using only the separate400-player validation cohort.
- Keep d96 and ResNet8c ensemble predictions, all game features, questions,
  expected-reward decoding and labels fixed.
- Report paired-player intervals with Bonferroni adjustment across the three
  leave-one-out contrasts. Do not choose a leave-one-out combination from
  holdout3 for production.
- A direction merits a fresh-cohort confirmation only if its adjusted
  interval lower bound is positive and the point gain is at least0.003.

This historical cohort has been explored previously. Results are a diagnostic
of seed robustness, not an independent estimate of future leaderboard score.
