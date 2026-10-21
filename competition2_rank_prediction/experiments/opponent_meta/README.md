# Score-raising experiments on top of best2 + opponent (2026-10-21)

All models are residual corrections z' = z + b + f(x) of the default pipeline's pooled logits (f = 0 reproduces the default),
decoded with the expected competition score. `run.py` (CV on the validation cohort, fresh cohort scored once, old holdout as
secondary), `scripts/extract_final_result.py` (KataGo final-position result proxy -> `features/final_result/{val,holdout,holdout2}`).
Full table: `logs/run.log`, `results/report.json`.

- E1 opponent distribution (mean / spread / quantiles / class means / agreement of the opponent member over the question's games)
- E2 player x opponent pairing (own expected rank mean / std, per-game own - opponent: std, quantiles, extremes, correlation)
- E3 game-result proxy: the SGF has no result, handicap or komi (no handicap openings and no recorded passes were found), so
  KataGo's final-position win rate / score lead for the player stands in for the result, plus an Elo-like performance
  (mean opponent rank + win share - 0.5). No game network: opponents are anonymous.

## Result: nothing confirmed, default unchanged
Pre-registered candidate (best validation CV): MLP residual on E1. Validation CV +0.014 [-0.006, +0.034]; fresh cohort
+0.000 [-0.030, +0.033]; old holdout +0.005 [-0.026, +0.037]. Not confirmed.

- No configuration beats the default with a validation-CV interval above zero; full linear residuals (10 x |x| weights) and
  gradient boosting overfit (validation -0.01 to -0.04, HGB -0.056 on the fresh cohort).
- Pattern worth noting (post hoc, 26 configurations looked at, so not evidence on its own): the 1-parameter-per-feature
  "shift" residual (moves the whole distribution along the rank axis) is positive on BOTH untouched cohorts for every feature
  set: fresh +0.006 .. +0.022, old holdout +0.015 .. +0.041, largest with the result proxy E3 (e.g. E2+E3, lam 0.01:
  fresh +0.012, old +0.041 [+0.015, +0.068]). Validation CV does not show it (-0.010 .. +0.010). Confirming it would need a
  third untouched cohort with one configuration fixed in advance.
