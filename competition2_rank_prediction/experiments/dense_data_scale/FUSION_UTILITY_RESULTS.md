# Bounded score-aware relative fusion: fixed screen fails

Completed2026-11-09 on the historical400-player validation cohort. Five
player-disjoint calibration folds,8,000shared questions. Original encoders
and C1 correction retained; only bounded relative member weights learned.

| Complete pipeline | Official utility |
| --- | ---: |
| Original C1+C2 | 0.7106057000 |
| Bounded question-NLL weight control | 0.7107507754 |
| Bounded smooth-utility weight fit | 0.7104807000 |

Utility−original:-0.00012500, paired-player97.5% interval
[-0.00057902,+0.00031606]. Utility−NLL:-0.00027008, interval
[-0.00098121,+0.00040803]. Both intervals include zero. The fixed screen
fails; no production change or additional bound/temperature/regularizer sweep.

Only6of8,000decisions change versus original:2improve,4worsen. Versus the
NLL control:17change,7improve,10worsen. The fitted weights remain close to
the starting values. This demonstrates little useful movement in this tested
local, regularized two-degree-of-freedom family, not global optimality of the
original weights or impossibility of all score-aware training.

## Correctness checks

An initial algebraic simplification stopped at fold4because it omitted the
production fused-probability floor before pooling. No completed result from
that attempt was accepted. The corrected differentiable implementation keeps
per-game log-softmax,1e-6floor and masked question means. The model objective,
training budget, folds and screen thresholds were not changed after failure.

All five zero-delta posterior errors are below1.38e-7and all zero-delta
decisions match. The original0.7106057000baseline replays exactly. New weights
are fit only on each fold's320players; the held80players never enter fitting.
Both objectives use300updates, same initialization, constraints and penalty.

Pylint10/10 and mypy pass for three new modules. Three tests cover weight
scale/bounds/gradients, both synthetic fitting objectives and exact pooling
with extreme probabilities plus padded query lengths. Outputs:
`production/fusion_utility_20261109/report.json` and `predictions.npz`.
The report records the input-cache SHA256 and all per-fold coefficients.

No GPU training, official-test inference, new submission or upload occurred.
This negative local screen does not resolve the complementary-error oracle:
no deployable selector has been established, and oracle0.772is not a score.
