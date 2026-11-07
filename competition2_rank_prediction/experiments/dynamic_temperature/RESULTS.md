# Dynamic temperature: calibration improves, utility does not

Completed 2026-11-07, continuing GitHub issue #1. This is an exploratory
result on the previously studied 2,000-player internal cohort, not a new
independent validation and not an official Public result.

## Frozen comparison

Same upstream-half-safe five-fold OOF protocol and expected-utility decoder
for all methods. The dynamic model fits only two parameters using one
prelocked uncertainty scalar: per-game consensus expected-rank standard
deviation divided by the square root of game count. It does not retrain
the encoders or change class ordering within each probability vector.

| Method | Competition utility | Exact | >=2-rank error | NLL | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen C1+C2 | 0.717012 | 58.20% | 5.10% | 1.07556 | 0.57100 | 0.12883 |
| Global temperature | 0.716960 | 58.25% | 5.20% | 1.02819 | 0.54848 | 0.04528 |
| Dynamic temperature | 0.716144 | 58.15% | 5.25% | 1.02133 | 0.54560 | 0.03714 |

Identity-temperature numerical and decoding parity passed; the global
control reproduced the previous utility exactly. Dynamic utility delta:

- Versus baseline: -0.000868; paired player-bootstrap 95% interval
  [-0.005293, +0.003506], 5,000 resamples.
- Versus global temperature: -0.000816; interval
  [-0.002712, +0.000948].
- Upstream-half deltas versus baseline: -0.003264 and +0.001528.
- 52/2,000 final decisions changed: 24 improved utility, 28 worsened it.
  Raw probability argmax did not change; utility decoding can change.

The preregistered interest gate failed. No fresh outer cohort, ordinal
scale head, production integration or new submission is scheduled for
this recipe. This is not proof that every uncertainty-aware method fails.

## What the audit shows

Uncertainty has real diagnostic information. The locked scalar correlates
with realized baseline utility at r=-0.190, and with leave-one-game-out
proxy decision instability at r=+0.383. Baseline utility in its lowest
and highest uncertainty quartiles is 0.7953 and 0.6208, respectively.
However, recognizing difficult questions does not identify which adjacent
rank should replace the current decision.

Across the fixed temperature grid [0.5, 2], 91% of questions retain the
same utility-decoded decision at every grid point. A label-informed grid
oracle reaches 0.74158, but is diagnostic only and is not a deployable
model or selection target. Scalar temperature cannot repair wrong class
ordering. About 23.95% of learned temperatures hit the lower bound;
the preregistered bounds were not retuned after this observation.

On the SAME 126 players with 20 games, repeated subset decision disagreement
with their full-set pooling proxy falls from 31.71% (5 games), to 24.09%
(8), 15.79% (12), and 9.29% (16). At 20 it is trivially zero because the
subset equals the full set. These are stability measurements of an
unweighted geometric ensemble proxy, NOT full C1+C2 scores or ground-truth
accuracy improvements. Repeated subsets are aggregated within player.

## Decision and next research implication

Keep C1+C2 as the production baseline. Improved NLL, Brier and ECE alone
are insufficient evidence of a competition-score gain. A future proposal
would need a mechanism that changes the relative evidence for neighboring
ranks, rather than another scalar confidence correction, and an untouched
outer evaluation before adoption. No such new experiment is claimed here.

Full subgroup, boundary-AUC, temperature and audit results are in the local
`production/dynamic_temperature_20261107/report.json`; reproduction details
are in [README.md](README.md).
