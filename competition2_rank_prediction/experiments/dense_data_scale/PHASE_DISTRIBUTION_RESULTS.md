# Fixed richer phase-profile forest: screen fails

Completed2026-11-09. Original400-player exploratory validation, own branch
only, not Public or complete C1+C2. Both arms trained on all15,211 original
training games with identical players, sample weights and classifier settings.

| Three-seed probability ensemble | Question utility |
| --- | ---: |
| Mean/std control,210dimensions | 0.5639761521 |
| Quantiles/skew,840dimensions | 0.5608725138 |

Rich-control delta-0.00310364, paired-player95% interval
[-0.00793684,+0.00146525]. Seed deltas+0.00099219,-0.00193394,-0.00492227.
One of three seeds improves. Worst rank delta3D -0.02520075. Query-size
deltas:5–9games-0.00436499,10–14-0.00570419,15–20+0.00019805.

The preregistered screen fails. Do not run a parameter/phase sweep or add this
branch to production on these results. Conditional full-pipeline evaluation
was gated and therefore not run. This does not show that all distribution
models are ineffective; it rejects this fixed matched-feature forest recipe.

Artifacts: `production/phase_distribution_20261109/`, six trained forest
pickles, aligned predictions and report. Numeric report arm keys2/8 are reused
scoring-helper identifiers, explicitly mapped in `arms`; they do **not** mean
two/eight games in this experiment.

Checks: pylint10/10 and mypy passed for all three new modules; two unit tests
cover masks/empty phases and expected moments/quantiles. No new submission,
GPU work or production changes. No breakthrough established.
