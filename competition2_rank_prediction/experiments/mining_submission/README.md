# Alternate rank-submission delivery: mining screen and joint candidate

Completed 2026-11-08. The upload-ready alternate is the joint candidate:
`production/rank_joint_20261108/submission_rank_joint_20261108.csv`.
See [the delivery record](../../docs/submission_joint_20261108.md).

The mining extension was attempted first. Validation-fitted fusion gave
it effectively zero weight, reproduced the stable decisions on all800
questions, and the delivery guard rejected that identical alternate. No
forced prediction changes were made. The predeclared fallback adds a
joint two-sided Transformer to C1+C2, with independent validation refitting.

## Inputs and outputs

- Original test ZIP: `data/aicup2026_rank_prediction_tests_20261030.zip`
  under the runtime research workspace, including Public400/Private400.
- Reuse its already extracted96-decision own/opponent/final features and
  move archive in `production/rank_20261030/`. No new KataGo extraction.
- The original `submission_rank.csv` remains byte-identical. Its exact
  replay is written separately as `baseline_reproduced.csv`.
- New output includes the CSV, manifest/checkpoint hashes, player-fold CV,
  probabilities, per-question method report and official-format validation.
  Models, data and generated CSVs stay outside Git in ignored directories.

## Reproduce locally

From the repository root:

```bash
export PYTHONPATH="$PWD/competition2_rank_prediction"
export OMP_NUM_THREADS=4
```

The existing joint training implementation was used read-only:

```bash
runtime/competition1_research/.venv/bin/python \
  runtime/competition2_research/experiments/joint_game/joint.py train \
  --tag joint_submit20261108 --seeds 3 --epochs 40 --batch 256 --device cuda:1
```

This command writes new checkpoint tags in the runtime results directory;
do not rerun it against existing tags unless intentionally regenerating
them. Training source SHA-256:
`43a843e9a0327be6059ab5b240ef8fba468cef74729f32ab7f03add1919f3c1e`.
The local runtime includes the historical experiment code and data; a
source-only clone needs those assets restored before training/inference.

Preparation and release (choose a fresh sibling output path for reruns):

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.mining_submission.prepare \
  --member joint --output production/rank_joint_20261108
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.mining_submission.run \
  --member joint --output production/rank_joint_20261108
runtime/competition1_research/.venv/bin/python \
  competition2_rank_prediction/scripts/check_submission.py \
  production/rank_joint_20261108/submission_rank_joint_20261108.csv \
  --tests-dir runtime/competition2_research/data/aicup2026_rank_prediction_tests_20261030/tests \
  --diff production/rank_20261030/submission_rank.csv \
  --json production/rank_joint_20261108/submission_validation.json
```

For the mining delivery screen use `--member mining` and a new output
directory. An identical output or a validation degradation over0.005 is
explicitly rejected. That screen has not been weakened for the joint run.

## Verification

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/mining_submission
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/mining_submission
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.mining_submission.test_delivery
```

Mypy verifies new modules with legacy/third-party dependency checking
limited by those flags. Tests cover side/time/pass representation and
reward-optimal CSV decoding, including invalid probability inputs.
The clean joint input implementation was also compared to the historical
implementation:32 actual games had exactly identical tokens and masks;
all8,000 validation game probabilities matched the training implementation
within2.4e-7. Official CSV validation covers the entire800-question output.
