# Dynamic uncertainty / temperature — issue #1

Exploratory Competition 2 experiment, completed 2026-11-07.
Production C1+C2 and official submission files remain unchanged.

Read [the preregistration](PREREGISTRATION.md) before
[the results](RESULTS.md). The experiment uses cached probabilities;
no GPU, KataGo extraction or upstream encoder retraining is needed.

From the repository root:

```bash
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python \
  -m competition2_rank_prediction.experiments.dynamic_temperature.run
```

The default output directory must not already exist. Use `--output` to
select a new run directory. Default artifacts (ignored by Git):

- `production/dynamic_temperature_20261107/report.json`: metrics,
  player bootstrap intervals, fold parameters, audits and stop decision.
- `production/dynamic_temperature_20261107/predictions.npz`: aligned
  player-level OOF predictions, temperatures and uncertainty diagnostics.
- `production/dynamic_temperature_20261107/temperature_diagnostics.png`:
  uncertainty versus temperature and temperature distributions.

The saved folds are read from the preceding rank-experts pilot. Cached
inputs and exclusion manifests are checked by its shared loader. Each
calibrator trains only within its upstream-excluded half, on other folds;
standardization is training-fold-only. Neither official tests nor Public
leaderboard scores enter training or selection.

Checks:

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/dynamic_temperature
runtime/competition1_research/.venv/bin/mypy \
  competition2_rank_prediction/experiments/dynamic_temperature
```
