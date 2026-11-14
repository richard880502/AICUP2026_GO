# Competition 2 current best

## Official Public result, 2026-11-06

The user reports a Public score of **0.7227214** for the 2026-10-30 test
submission. All 800 Public and Private questions used the default C1+C2
pipeline, with no fallback. The Private score has not yet been disclosed.
See [the submission record](docs/public_submission_20261030.md).

Last updated: 2026-11-10.
The completed [expanded d96 training experiment](experiments/dense_encoder_scale/RESULTS.md)
scored 0.714955 versus original C1+C2 0.717012 on the historical 2,000-player
cross-fit. The paired delta was -0.002057 (95% interval [-0.007298, +0.003000]);
both folds regressed. Keep the original production model and submission.

The user reports **0.7195608** for the Joint alternate submission, versus
the original **0.7227214** (delta **-0.0031606**). Joint is not adopted;
the original C1+C2 remains the Public-best production method.
The completed [reward-alignment experiment](experiments/reward_alignment/RESULTS.md)
improved exploratory calibration CV from original0.710606 to0.712950,
but the paired interval versus original crosses zero. Original C1+C2
remains production; no reward-aligned alternate submission was generated.
The internal C1+C2 experiment was registered and evaluated on 2026-10-21.

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

## DEFAULT since 2026-10-21: best2 + opponent-specific member (C2) + shift residual (C1)
Pre-registered 2,000-player cross-fitted test (`experiments/crossfit_cohort`): default 0.6763; C1 shift residual 0.7061
(+0.030 [+0.017, +0.043]); C2 opponent-specific Transformer 0.6833 (+0.007 [+0.001, +0.013]); both pass Holm.
C1 + C2 0.7170 (+0.041, exploratory). Integrated into `predict_test.py` as the default on 2026-10-21 at the user's request
(`residual_shift.py`; member `opp_scratch`); the integrated script reproduces the evaluation exactly (halves 0.7265 / 0.7075).
Switches: `--no-residual` (offsets instead of C1), `--opp-model d96` (borrowed member instead of C2), `--no-opp`.
Test day: `scripts/run_test_extraction.sh` now also writes `features/final_result/test` (needed by C1).

The tracked extraction, SGF validation, prediction and submission-checking
code has been synchronized with the local workspace that produced the official
submission. See [recent experiment findings](docs/recent_experiments.md) for
the 2026-10-22 through 2026-11-04 investigations and their limitations.

## Previous default (2026-10-20): best2 + opponent-side member

The 2026-10-12 "opponent side rejected" result was a bug (`extract_rank_features.py --opponent` ignored the flag). With the fix,
the production d96 Transformer applied unchanged to the OPPONENT's 96 decisions of each game, log-fused as a third member
(weights + offsets fitted on validation only), gives:

| Method | 400-player holdout | validation questions (8,000, 5-fold CV weights/offsets) |
| --- | ---: | ---: |
| best2 (default) | 0.6609 | 0.6350 |
| **best2 + opponent** (`predict_test.py --opp-feat ...`) | **0.6808** | **0.6875** |
| d96 + opponent (no ResNet) | 0.6837 | 0.6858 |
| opponent member alone | 0.6738 | 0.6834 |

Paired differences vs best2: holdout +0.020 [-0.003, +0.043], validation +0.053 [+0.040, +0.065] (player bootstrap).
d96 + opponent vs d96 alone: holdout +0.040 [+0.012, +0.068], validation +0.061 [+0.046, +0.076].
This is the first direction with a validation interval far from zero; the holdout has been looked at before, so validation is the
main evidence. Likely mechanism: opponents are matched by the *label* rank, so averaging different opponents over 5-20 games tracks
the label, while the player's own moves carry a persistent player-specific offset (see `experiments/a1d_boundary`).
Test-day cost: one extra `extract_rank_features.py --opponent` pass on the test shards (same time as the player pass).
**Fresh-cohort confirmation (2026-10-20, `experiments/fresh_cohort`):** 400 new players, models `*_h2` retrained without them:
best2 0.6535 -> best2 + opponent 0.7206, +0.067 [+0.046, +0.090]. Made the `predict_test.py` default on 2026-10-20 at the user's request
(`--no-opp` = previous best2). Test day: `scripts/run_test_extraction.sh` extracts both passes. Details: `experiments/opponent_side/README.md`.

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
