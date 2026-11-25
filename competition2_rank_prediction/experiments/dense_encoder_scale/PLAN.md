# Dense per-game encoder scaling experiment

Status: completed; the full-pipeline advancement gate failed. See
[RESULTS.md](RESULTS.md). The protocol below records the evaluated settings.

## Hypothesis

The production d96 per-game Transformer sees about two games per training
player. An already extracted 36,000-game cohort provides about 18 additional
games for each of 2,000 balanced training players. Adding these examples to
the original d96 training data may improve the game-level rank signal and the
full C1+C2 rank-prediction pipeline.

This differs from the previous frozen-embedding player-bag experiment: the
d96 encoder itself is trained on the additional game-level features.

## Data and leakage controls

- Base features: 15,211 games from 7,607 training players.
- Additional features: 36,000 games from 2,000 training players.
- The game-key intersection is empty.
- Validation-player intersection is empty.
- The 2,000-question holdout3 cohort is split into two player-disjoint folds.
- Each fold excludes its own 1,000 holdout3 players from the merged training
  features. The observed training sizes after exclusion are 42,227 and 42,587
  games.
- Validation is used for fusion and C1 fitting. Holdout3 is evaluated once
  after all six models are trained. No official Public labels or feedback are
  used in this experiment.

## Training and evaluation

Each fold trains the existing d96 recipe with three seeds, 40 epochs, and the
existing player-disjoint inner split for checkpoint selection. Training uses
CPU because GPU memory is reserved by other services. Checkpoints are written
to the ignored runtime research directory and predictions to the experiment
output directory.

For each fold, the candidate d96 member is added to the original d96, ResNet,
and opponent-specific C2 members. Fusion weights and the C1 residual are fit
on validation only. The primary metric is the paired per-question competition
utility on the 2,000-player cross-fit cohort.

Advance only if all three conditions hold:

- Utility gain is at least +0.003.
- The paired player bootstrap 95% interval has a lower bound above zero.
- Both cross-fit fold deltas are non-negative.

Otherwise, keep the original C1+C2 production model and submission.

## Interpretation limits

The extra cohort has exactly 3,600 games per rank. The base pool is less
balanced (for example, A has 3,083 games and 1D has 785). Adding games changes
both within-player coverage and the class mixture, and gives the selected
2,000 players more weight than the remaining players. A positive result
would establish a candidate pipeline gain, but would not isolate coverage as
the cause without a matched sampling/weighting control. Per-rank input counts
are saved in `production/dense_encoder_scale_20261109/rank_mix.json`.

## Artifacts

- Merged, fold-excluded feature symlinks:
  `production/dense_encoder_scale_20261109/merged_train_c{0,1}/`
- Training checkpoints:
  `runtime/competition2_research/experiments/dense_encoder_scale/results/`
- Per-game predictions and final report will be written under:
  `production/dense_encoder_scale_20261109/`
- Shared fold-aware scoring and evaluation entry points:
  `experiments/full_train_coverage/score.py` and `evaluate.py`

The evaluation entry points accept an alternate tag prefix and candidate
name; their defaults preserve the existing full-training-coverage experiment.

After all six checkpoints exist, run from the repository root:

```bash
export PYTHONPATH="$PWD/runtime/competition2_research"
python=runtime/competition2_research/c1/.venv/bin/python
for fold in 0 1; do
  "$python" -m competition2_rank_prediction.experiments.full_train_coverage.score \
    --fold "$fold" --device cpu --tag-prefix d96_densebag \
    --checkpoint-dir runtime/competition2_research/experiments/dense_encoder_scale/results \
    --probability-dir production/dense_encoder_scale_20261109/probabilities
done
"$python" -m competition2_rank_prediction.experiments.full_train_coverage.evaluate \
  --candidate-name densebag \
  --probability-dir production/dense_encoder_scale_20261109/probabilities \
  --output-report production/dense_encoder_scale_20261109/crossfit_report.json
```

Scoring requires seeds 0, 1, and 2 for the selected fold and refuses an
incomplete ensemble. The evaluator explicitly uses CPU for legacy opponent
checkpoint inference.

`finish.sh` performs these steps after optional live training PIDs terminate:

```bash
bash competition2_rank_prediction/experiments/dense_encoder_scale/finish.sh \
  <fold0_training_pid> <fold1_training_pid>
```

It requires all six checkpoints and both final validation reports before
starting inference. Its log is `logs/postprocess.log` under the production
experiment directory; successful completion writes `postprocess_completed.txt`.
Do not launch another copy while this postprocessing job is active.
