# Why the frozen experts lost utility

Completed 2026-11-07, following the updated issue #1. Analysis-only follow-up
on the previously studied 2,000-player cohort; no new production candidate.
Original pilot predictions were refitted EXACTLY (maximum probability
difference 0 for bias, linear, region and boundary models). GPU 1 runtime
was about 11 seconds; no KataGo extraction or upstream retraining.

## P0: train vs OOF

Ten equally sized folds, each with 800 training and 200 evaluation players.
Training summaries reuse each player four times and are descriptive only.

| Model | Train utility at epoch 0 | Train utility at epoch 200 | OOF utility at epoch 200 |
| --- | ---: | ---: | ---: |
| Linear | 0.71701 | 0.77205 | 0.69979 |
| Region | 0.71701 | 0.83677 | 0.69568 |
| Boundary | 0.71701 | 0.73569 | 0.70771 |

The region training NLL drops from 1.07556 to 0.76181. OOF NLL improves
initially to 1.03584 at epoch 50, then worsens to 1.04659 at epoch 200, while
OOF utility declines from 0.71701 to 0.69568. Even the fixed epoch-20
snapshot has no utility gain (0.71575). These curves provide evidence of
overfitting; the negative outcome is not just failure to fit the training
set. Epochs were observed for diagnosis only, not selected for deployment.

## Calibration is a different objective

NLL clips probabilities at 1e-9 for every method, matching pilot input logits.
Brier is the sum of ten squared class errors. ECE is pooled 15-bin top-label
calibration, using raw probability argmax; utility uses expected-score decode.

| Method | OOF utility | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| Frozen baseline | 0.71701 | 1.07556 | 0.57100 | 0.12883 |
| Linear | 0.69979 | 1.05394 | 0.56568 | 0.01588 |
| Region | 0.69568 | 1.04659 | 0.56234 | 0.02826 |
| Boundary | 0.70771 | 1.07147 | 0.56900 | 0.10080 |
| One-parameter temperature control | 0.71696 | 1.02819 | 0.54848 | 0.04528 |

Temperature is trained within folds only, with no leaderboard use. Improved
NLL, Brier or ECE does not translate into utility gains here. This supports
objective mismatch as an additional factor, without proving CE is universally
unsuitable. Temperature changes 2.25% of decisions, with nearly balanced wins
and losses. Region changes 21%: 8.45% improve utility, 11.95% worsen it.
Baseline mean entropy is 1.24387; region is 1.10919. Region centered log
correction RMS is 0.25931, with absolute q90 of 0.45256.

## P0: where utility is lost

| True rank | Region delta | Boundary delta |
| --- | ---: | ---: |
| D | -0.02345 | 0 |
| C | -0.01264 | 0 |
| B | +0.02345 | 0 |
| A | +0.00684 | +0.01000 |
| 1D | -0.06321 | -0.05373 |
| 2D | -0.04212 | -0.02396 |
| 3D | +0.00316 | -0.00948 |
| 4D | -0.06373 | -0.00632 |
| 5D | -0.02080 | -0.01264 |
| 6D | -0.02080 | +0.00316 |

Region losses are not confined to A/1D: 1D and 4D account for about 60% of
the net overall loss (each rank has 200 players). For 5-9 / 10-14 / 15-20
games, region deltas are -0.02914 / -0.00743 / -0.02694; boundary deltas are
-0.01062 / -0.00866 / -0.00874. Both upstream halves lose utility.

| Conditional pair-odds AUC | Baseline | Region | Boundary |
| --- | ---: | ---: | ---: |
| A/1D | 0.81355 | 0.79433 | 0.79230 |
| 1D/2D | 0.72573 | 0.70615 | 0.72325 |
| 4D/5D | 0.87650 | 0.85113 | 0.87118 |

These full-C1+C2 pair-odds AUCs on this cohort are NOT directly comparable
to the older scalar-strength AUC 0.609 from a different model and cohort.

## P1: redundant features and a limited new hypothesis

Fold-safe ridge reconstruction of 79 extra features from the ten frozen
centered logits has median held-out R2 of 0.69668 (IQR 0.43362-0.86994).
Opponent mean strength is almost reconstructible (R2 0.99130), consistent
with C1/C2 already using it. This is linear redundancy, not proof that all
nonlinear or conditional information has been exhausted.

The largest absolute correlations of residualized features with rank error
are only about 0.12. One is final-winrate standard deviation (r=0.11904,
reconstruction R2=-0.00520). It is weak and selected from 79 comparisons;
do not interpret it as validated incremental evidence.

Independently testable hypothesis for a possible later experiment: question-
level outcome volatility may add a small correction beyond C1's outcome
means. A valid test would preregister ONE scalar shift term using
`std(final_winrate)`, retain baseline and temperature controls, and confirm
on a new player-disjoint cohort after all weights are frozen. The current
region model already received this input and lost utility, so the observation
does not justify another large MLP or a production change. No such candidate
was trained or selected in this diagnosis.

## Decision

Stop expansion of this frozen MLP expert recipe. Do not build learned gates,
combine experts or enlarge the OOF training pool merely to rescue it.
More OOF data would require upstream models excluding the entire new expert
development/evaluation pool; concatenating the present halves is unsafe.
That includes new d96, opponent and ResNet training, so it is not a free
cache enlargement. Current evidence does not justify that expense.
Production remains C1+C2, user-reported official Public 0.7227214.

Reproduce from repository root:

```bash
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.rank_experts.diagnose
```

The command refuses to overwrite existing output. Results are under
`production/rank_experts_diagnostics_20261107/`: `report.json` contains all
fold snapshots, subgroup metrics and named feature diagnostics;
`training_curves.png` visualizes train/OOF divergence. Preliminary debug
outputs are retained separately; only this final directory is authoritative.
