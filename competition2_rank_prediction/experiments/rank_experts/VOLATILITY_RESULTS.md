# Final-winrate volatility: weak association, no utility improvement

Completed 2026-11-07 under `VOLATILITY_PLAN.md`. This is a targeted follow-up
to a feature selected from 79 columns on a previously studied cohort.
All associations and intervals below are exploratory, not an independent
confirmation or a justification for production adoption.

## What was found

The feature is the population standard deviation of KataGo final-position
player winrates across games of one question. It is a model-based terminal
estimate, not an observed match result. Its association with the frozen
model's rank error reproduces in both upstream-excluded halves:

| Group | Players | Raw volatility vs rank error r | Conditional volatility vs rank error r |
| --- | ---: | ---: | ---: |
| All | 2,000 | 0.12070 | 0.12606 |
| Half 0 | 1,000 | 0.07739 | 0.07345 |
| Half 1 | 1,000 | 0.15434 | 0.16947 |

Rank error is true rank minus baseline expected rank. A positive value means
larger volatility tends to accompany underestimation. Conditional volatility
uses training-only ridge residualization against the ten centered frozen
logits, mean terminal winrate, its square, and log game count. This is a
specified linear nuisance model, not proof of conditional independence.
Ten fitted primary coefficients are all positive (0.03266-0.07126).

The outcome-balance controls explain 81.69% of volatility's held-out
variance. Therefore much of the feature is related to outcome balance,
and it should not be presented as a wholly new independent source of rank
information. The direct terminal-undecidedness statistic E[w(1-w)], derived
from var(w)=E[w]*(1-E[w])-E[w(1-w)], has essentially zero rank-error
correlation (r=0.00149). The original weak association alone cannot establish
that terminal-estimation uncertainty is the useful mechanism.

## Can the association improve predictions?

One question per player; identical five folds within each upstream-excluded
half. The primary bounded scalar rank-axis shift has one volatility weight
and one intercept, with fixed CE/L2 training. No hyperparameter search,
KataGo extraction, encoder retraining or official test data was used.

| Method | OOF utility | Exact | Error >=2 ranks | NLL |
| --- | ---: | ---: | ---: | ---: |
| Frozen C1+C2 | 0.717012 | 0.5820 | 0.0510 | 1.075561 |
| Axis-bias-only control | 0.714115 | 0.5780 | 0.0520 | 1.075285 |
| Raw-volatility shift (secondary) | 0.714299 | 0.5780 | 0.0515 | 1.074706 |
| Conditional-volatility shift (primary) | 0.714299 | 0.5780 | 0.0515 | 1.074663 |

Primary utility delta vs baseline: **-0.002712**, paired-player 95% interval
**[-0.007637, +0.001896]**. Against bias-only: +0.000184
[-0.001396, +0.001764]. Both half utility deltas are negative:
-0.003264 / -0.002161. The fixed exploratory-interest gate fails.

There is a small probabilistic increment relative to bias-only: NLL delta
-0.000622 [-0.001120, -0.000203] and Brier delta -0.000187
[-0.000326, -0.000052]. These nominal intervals are descriptive after
feature selection, and do not establish improved competition utility.
Against the original frozen baseline, both loss intervals include zero.

The primary changes 66 of 2,000 decisions (3.3%): 28 increase utility, 37
decrease it, and one changes rank without changing utility. Gains at D/C/B
are offset particularly by 1D/2D losses (-0.01580 / -0.01448 within those
ranks). Each game-count group has a negative delta (-0.00408 / -0.00098 /
-0.00308 for 5-9 / 10-14 / 15-20 games).

## Decision and evidence limits

The requested weak association was located and reproduced descriptively in
both halves, but its predictive usefulness is NOT confirmed. Keep production
C1+C2 (user-reported Public 0.7227214). Do not claim an official score gain,
submit a replacement, or expand this model on the basis of these results.
Original feature selection and this test share the same cohort; even though
all fitted corrections are player-held-out, neither half is now untouched.

Reproduction from repository root:

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.rank_experts.volatility
```

The command refuses to overwrite completed output. Full audit, coefficients,
associations, all rank/count deltas and paired bootstrap intervals are in
`production/final_winrate_volatility_20261107/report.json`; aligned OOF
probabilities and player/fold IDs are in `predictions.npz`. All data and
player-level artifacts remain ignored by Git. The baseline score is exactly
0.7170117549; the two-half manifests, labels and player ordering were checked.
