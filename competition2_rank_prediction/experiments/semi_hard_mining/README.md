# Ordinal pair-loss and conditional semi-hard mining

Completed A/B screening on 2026-11-08. See
[PREREGISTRATION.md](PREREGISTRATION.md), [RESULTS.md](RESULTS.md) and the
initial [data inventory](PLAN.md). The conditional C mining arm did not
qualify to run under that first protocol. The later user-requested direct
mining experiment is documented below; the original stop remains recorded.

From the repository root:

```bash
export PYTHONPATH="$PWD/competition2_rank_prediction"
export OMP_NUM_THREADS=4
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.semi_hard_mining.test_losses
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.train \
  --device cuda:1
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.evaluate \
  --device cuda:1
```

Choose a new `--output` directory for a rerun; existing runs are protected
against accidental replacement. All local data and checkpoints remain in
ignored `runtime/` and `production/`, outside Git. Default output:
`production/semi_hard_mining_20261108/`.

Artifacts include `data_audit.json`, `splits.npz`, A/B selected checkpoints,
training history, per-player inner/evaluation predictions, `report.json`
and `training_curves.png`. The initial report before adding selected-model
gradient diagnostics is retained as `report_initial.json`; the decision
metrics are exactly identical between these reports.

Checks:

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/semi_hard_mining
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/semi_hard_mining
```

The mypy flags scope the check to the new modules; existing legacy imports
and sklearn do not provide complete type information. They do not verify
the internals of those dependencies. No Python lint-disable comments were
added. The legacy Transformer module needs the PYTHONPATH setting above.

## Second-round three-seed replication

Completed; see [MULTISEED_RESULTS.md](MULTISEED_RESULTS.md). The fixed
mining gate failed despite a positive evaluation utility point estimate.

The user requested continuation after the initial screen. The separately
fixed [multi-seed protocol](MULTISEED_PREREGISTRATION.md) increases the
shared A/B budget to60 epochs x64 updates, using three new seeds and all
original player splits. Start from the same repository root and venv:

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.multiseed_train \
  --device cuda:1
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.multiseed_evaluate \
  --device cuda:1
```

The default output is `production/semi_hard_mining_multiseed_20261108/`.
For another run choose a fresh sibling directory with `--output`, retaining
the first-round output in the same parent for split verification. Read
`protocol.json` for seeds, budget and source/preregistration fingerprints;
each `seed_<seed>/` stores A/B checkpoints and histories. Evaluation writes
`report.json`, per-player seed prediction arrays and
`multiseed_training_curves.png`. The primary interval resamples 300 players
after averaging their paired seed differences, not 900 predictions.

Run the loss and replication tests with:

```bash
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.semi_hard_mining.test_losses \
  competition2_rank_prediction.experiments.semi_hard_mining.test_multiseed
```

## User-requested direct mining C

Completed after a separate explicit request to test mining despite the
previous gate. Read [DIRECT_MINING_PREREGISTRATION.md](DIRECT_MINING_PREREGISTRATION.md)
and [DIRECT_MINING_RESULTS.md](DIRECT_MINING_RESULTS.md). The fixed C-B
comparison does not establish incremental mining benefit.

Using the same PYTHONPATH and venv setup:

```bash
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.semi_hard_mining.test_losses \
  competition2_rank_prediction.experiments.semi_hard_mining.test_multiseed \
  competition2_rank_prediction.experiments.semi_hard_mining.test_miner
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.direct_mining_train \
  --device cuda:1
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.semi_hard_mining.direct_mining_evaluate \
  --device cuda:1
```

Output defaults to `production/semi_hard_mining_direct_20261108/`; keep
the fixed multi-seed A/B controls in the sibling directory. Existing output
is protected from overwrite. C's three checkpoints and histories are new;
the A/B checkpoints are read-only and fingerprinted in the protocol.
