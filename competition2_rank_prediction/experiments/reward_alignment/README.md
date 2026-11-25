# Official-reward alignment

Isolated experiment based on the original C1+C2 production method. The
own-side d96 Transformer is retrained; ResNet and opponent-specific C2 are
frozen. C1 and fusion are refitted only within calibration-training folds.
This is **not** Joint or mining. See [fixed plan](PLAN.md).
Completed findings: [RESULTS.md](RESULTS.md).

## Commands

From repository root, with `PYTHONPATH=competition2_rank_prediction` and
`OMP_NUM_THREADS=4`, use the project Python interpreter:

```bash
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.reward_alignment.train
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.reward_alignment.audit
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.reward_alignment.evaluate
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.reward_alignment.conclusions
```

Default artifact directory: `production/reward_alignment_20261108/`.
Training/evaluation refuse to overwrite completed artifacts. Inputs are
read-only under `runtime/competition2_research/`. No SGF re-extraction,
test-label access, test predictions or automatic Aidea upload is needed.

Outputs: per-seed checkpoints, histories, immutable split player lists,
source-hash manifest, `audit.json`, `learning_curves.png`,
`evaluation.npz`, and `report.json`.

Training uses two loss runs per seed. Each saves both inner-NLL and
inner-question-utility checkpoints. The three planned arms are NLL/NLL,
NLL/utility and mixed/utility, all with identical architecture and budget.
The reward/NLL artifact is diagnostic only, not a fourth model selection.
Seed-level diagnostics reuse each arm's ensemble-fitted calibration;
they are not separately optimized per-seed production pipelines.

Normalization deliberately uses optimizer rows only (the historical
trainer normalized the whole training library before its inner split).
The matched NLL control distinguishes reward effects from this hygiene
change. Supplementary dense games are for inner selection only; sparse
inner players still contribute NLL, but utility uses only >=20-game
inner players. No dense games of optimizer players are added.

The outer cohort was used in historical investigations. Cross-validation
isolates calibration fitting from evaluation but cannot erase adaptive
research reuse. Confidence intervals are exploratory, not a substitute
for fresh-player confirmation or official Public measurement.

## Checks

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/reward_alignment
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/reward_alignment
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.reward_alignment.test_objective
```

Mypy checks changed source; imported legacy modules and missing third-party
stubs are outside this scoped check. No pylint warning suppressions added.
