# Full-player training coverage — no gain; the gate fails

Completed 2026-11-17. Historical 2,000-player cross-fit, labels used only for
final screening. Not a Public result and not a new submission.

The extraction that [PLAN.md](PLAN.md) prepared on 2026-11-09 but could not
start — both GPUs were occupied, and the CPU fallback implied about 30 days —
was run once GPU 1 became free. It took **about four hours**, not 30 days:
six workers on one 3090 sustained 5.8 games/s over the 45,013 prepared games,
and all 91 shards extracted with no missing or failed shard.

## Result

| Condition | Utility |
| --- | ---: |
| Original C1+C2 baseline | 0.7170118 |
| Expanded-coverage candidate | 0.7164315 |

```text
delta        -0.00058
paired player bootstrap 95% CI   [-0.00329, +0.00205]
cross-fit fold deltas            -0.00153 / +0.00037
advancement gate (>= +0.003, interval above zero, both halves non-negative)
                                 FAILED
```

The baseline replays 0.7170118, which matches the C1+C2 number measured
independently in this session from the cached opponent-identity inputs
(0.7170). Both halves do not improve; one regresses.

## Reading

Each training player went from about two games to as many as eight, and the
dense rank encoder got very slightly **worse**. Combined with
[strength_score_v2](../strength_score_v2/RESULTS.md), where a differently
trained scalar over a richer representation correlated 0.957 with what the
pipeline already computes and earned a fusion weight of exactly zero, the
own-move channel looks **information-saturated rather than data-starved**.
More games per player do not add rank information that C1+C2 is missing.

This is the same conclusion the `style_knn` probe reached from the other
direction: a style encoder never trained on rank scores 0.653 alone yet adds
-0.001 on top of C1+C2.

## Isolation

- Input selection is the fixed 2026-11-09 sample: seed 20261109, six unseen
  games for 7,348 players, all remaining games for 259 players, ten players
  with no unused source games, zero game-key duplicates against the existing
  training features, zero validation-player overlap.
- Extraction used the unchanged `extract_rank_features.py` with 96 samples per
  game and profiles `rank_5k,rank_1d,rank_5d`, physical GPU 1.
- Training merged base and added features by symlink into
  `merged_train_c{fold}`, excluded that half's players, and used the same
  three-seed d96 Transformer and 40 epochs. Original checkpoints were not
  overwritten.
- Evaluation cohorts and the question set are unchanged.

## Decision

Keep the current production model and submission. No official-test inference,
no new submission, no upload. Artifacts are under
`production/full_train_coverage_20261109/`.
