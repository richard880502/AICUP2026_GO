# Competition 2 current best

Last verified: 2026-10-07 UTC.

## Stable production pipeline

```text
SGF
  -> up to 96 player decision points
  -> KataGo normal policy + HumanSL 5k / 1d / 5d
  -> 3-seed ordinal Transformer
  + cosine-schedule ResNet snapshot ensemble
  -> validation-fitted geometric fusion
  -> log-mean question pooling
  -> validation-fitted class offsets
  -> expected competition-score decoding
```

`predict_test.py` uses this two-model set by default. The previous four-model
pipeline remains available with `--model-set full4`.

## Verified scores

| Method | 400-player holdout score |
| --- | ---: |
| Tutorial reproduction | 0.4031 |
| Historical four-model production pipeline | 0.6561 |
| **Selected two-model pipeline** | **0.6609** |
| Dense player-bag + rank-logit candidate | **0.6723** |
| **Cross-feature dense player-bag candidate** | **0.6780** |

The leading candidate fuses two independently trained cross-game
Transformers: the original 414D rank-logit representation and a 606D version
that also contains a frozen 192D raw-move embedding. The validation-locked
move-bag weight is 0.30, followed by a validation-locked 0.28 blend with the
stable production pipeline. Its locked holdout score is 0.677955, an observed
+0.017098 lift over production and +0.005661 over the earlier 0.672294 dense
candidate. The paired 95% intervals are `[-0.008708, +0.042902]` versus
production and `[-0.009482, +0.021464]` versus the earlier candidate. It is
the current best *candidate*, but the intervals cross zero and therefore a
fresh outer cohort is still required for confirmation.

Two-model minus four-model is `+0.00474`; paired bootstrap 95% interval is
`[-0.00316, +0.01264]`. The observed gain is not statistically conclusive,
but five-fold player-disjoint question-level fusion consistently assigns
near-zero useful weight to the removed move model and tutorial ResNet. The
two-model default is also faster and has fewer failure points.

## Reproduce the final comparison

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.model_selection.evaluate
```

End-to-end holdout submission rehearsal:

```bash
c1/.venv/bin/python predict_test.py \
  --tests-dir features/holdout/tests \
  --feat features/feat96/holdout \
  --moves features/moves/holdout_moves.npz \
  --suffix _h \
  --expect 400 \
  --labels features/holdout/tests/labels.csv \
  --out experiments/model_selection/results/submission_best2.csv
```

This command was run successfully and reproduced `0.6609` on 400 questions.

Reproduce the dense candidate's locked comparison (no hyperparameter search):

```bash
c1/.venv/bin/python -m experiments.dense_player_bags.evaluate_holdout \
  --embeddings experiments/dense_player_bags/embeddings_logits \
  --checkpoint experiments/dense_player_bags/results/transformer_logits/model.pt \
  --blend-weight 0.34 --device cpu
```

Reproduce the current cross-feature candidate (all weights are locked and the
script performs no holdout search):

```bash
c1/.venv/bin/python -m \
  experiments.dense_player_bags.evaluate_cross_feature_holdout \
  --device cpu
```

## Experiment status

- `experiments/model_selection/`: selected result and paired uncertainty.
- `experiments/question_fusion/`: low-capacity question fusion improved CV
  from 0.6312 to 0.6383 but scored only 0.6523 on holdout; rejected.
- `experiments/bag_distribution/`: richer nonlinear bag statistics failed to
  generalize; rejected.
- `experiments/backbone_scaling/`: longer, larger, and more regularized dense
  Transformers did not beat the three-seed d96 ensemble; rejected.
- `experiments/robust_pooling/`: repeated player-disjoint CV retained standard
  log-mean pooling (0.6324 versus 0.6307 for the closest trimmed variant);
  rejected without reading holdout.
- `experiments/dense_player_bags/`: baseline frozen embeddings were rejected
  (0.660312 versus 0.660856); adding frozen per-game rank-logit features gave
  the 0.672294 candidate. The raw-move model alone was weaker, but combining
  its complementary predictions with the rank-logit bag improved two fixed
  validation samplings and reached the current 0.677955 locked holdout
  result. Size-conditioned blending, extra bag seeds, and raw decision
  statistics were rejected. The candidate needs fresh-cohort confirmation.
- `archive/`: recoverable superseded checkpoints and ablation artifacts.

The next trustworthy gain must be confirmed on a fresh player-disjoint outer
cohort because the current holdout has now been used for method comparison.
