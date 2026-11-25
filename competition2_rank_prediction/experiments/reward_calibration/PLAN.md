# Official-reward calibration objective screen

## Question

Does replacing only the C1 question-level residual's cross-entropy objective
with a fixed mixed objective improve held-player official utility?

The mixed loss is categorical cross-entropy plus `0.2 * (1 - expected
official reward)`. The official reward table is exact rank `1`, adjacent rank
`exp(-1)`, and otherwise `0`. The residual architecture, input features,
initialization, standardization, optimizer, learning rate, 300 steps,
regularization, fusion weights, game probabilities, question samples and
outer folds remain unchanged.

## Evaluation

- Use the original 400-player / 8,000-game cached validation cohort only.
- Reuse the fixed five player-disjoint folds and 20 fixed 5-20-game queries
  per held player.
- Fit fusion and the residual using only the other 320 players in each fold.
- Compare original NLL C1 against mixed-objective C1 using the same frozen
  three members (`d96`, `resnet8c`, `opp_scratch`).
- Report official utility, exact accuracy, NLL, rank and query-size deltas,
  and paired-player 97.5% bootstrap intervals.
- No coefficient search, additional seeds, Public inference, or submission
  generation. Keep the experiment only if the interval lower bound is above
  zero and the mean gain is at least `0.003`.

This is an exploratory calibration-CV screen, not an independent Public
estimate. It does not authorize replacing the preserved production CSV.
