# Dense player-bag experiment

Goal: apply direct 5-20 game player training to the strongest 96-decision
KataGo/HumanSL representation.

The targeted cohort contains 200 players per rank and 20 games per player.
Existing two-game dense features are reused; only the missing games are sent
through KataGo. Fixed validation and holdout players are excluded.

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.dense_player_bags.make_subset
bash experiments/dense_player_bags/run_extraction.sh
```

Extraction is resumable because completed `feat_*.npz` shards are skipped.
The launcher defaults to twelve workers (six per RTX 3090); set
`NUM_WORKERS` explicitly when using a different GPU memory budget.

After extraction finishes, create frozen 3-seed d96 game embeddings, append
frozen per-game ordinal rank-logits, and train the direct player-bag model:

```bash
c1/.venv/bin/python -m \
  experiments.dense_player_bags.extract_game_embeddings --device cuda:0
c1/.venv/bin/python -m \
  experiments.dense_player_bags.attach_game_logits
c1/.venv/bin/python -m \
  experiments.dense_player_bags.train_bags --device cuda:1
```

The final comparison trains both the cross-game Transformer and a lower
variance DeepSets control. The default uses 32 fresh bag batches per epoch;
the earlier six-step schedule under-sampled the 1,700 training players.

Run both controls without overwriting either checkpoint:

```bash
DEVICE=cuda:0 bash experiments/dense_player_bags/run_training.sh
```

Use `DEVICE=cpu CPU_THREADS=4` when the GPUs are reserved by another service.

Compare each checkpoint with production on aligned external-validation
questions without reading holdout:

```bash
c1/.venv/bin/python -m \
  experiments.dense_player_bags.evaluate_validation \
  --checkpoint experiments/dense_player_bags/results/deepsets/model.pt \
  --device cuda:0
```

## Result

The 36,000 newly extracted games plus 4,000 existing games produced a
2,000-player, 20-game-per-player training cohort. A 384D frozen three-seed
`d96_h` game embedding was used for a DeepSets and a cross-game Transformer.

| Model | External validation | Validation blend | Locked holdout blend |
| --- | ---: | ---: | ---: |
| Production | 0.6330 | 0.6330 | **0.660856** |
| DeepSets | 0.6282 | 0.6424 | not promoted |
| Transformer, 384D frozen embedding | 0.6438 | **0.6486** | 0.660312 |
| Transformer, 384D + 30 frozen rank-logits | 0.6436 | **0.6541** | **0.672294** |
| Transformer, rank-logits + 114 decision statistics | 0.6363 | 0.6478 | not read |
| Transformer, rank-logits + 192 raw-move embedding | 0.6282 | 0.6504 | 0.633331 alone |
| **414D + 606D bag prediction fusion** | complementary blend | **0.6574 / 0.6595 on two samplings** | **0.677955** |

The plain-embedding Transformer blend used validation-selected bag weight
0.24. Its holdout delta was -0.000544 with paired 95% interval
[-0.025857, 0.024339], so it is rejected.

The rank-logit version appends the 10-class log-probabilities from each frozen
`d96_h` ordinal head (30 values) to the 384D embedding. This requires no new
KataGo evaluation. Its validation-selected blend weight was 0.34 and its
single locked holdout score was 0.672294, a +0.011438 observed lift over
production. The paired 95% interval [-0.018847, 0.041321] crosses zero;
therefore it is the leading candidate, not a statistically confirmed
replacement.

The 606D raw-move branch is weaker in isolation, but is complementary to the
414D bag. The move-bag weight 0.30 and final production-to-bag weight 0.28
were selected on validation seed 20261009, then evaluated unchanged on a
second validation sampling (seed 20261019). The locked holdout score is
0.677955: +0.017098 over production and +0.005661 over the 414D candidate.
The respective paired 95% intervals are [-0.008708, 0.042902] and
[-0.009482, 0.021464], so fresh-cohort confirmation remains necessary.

```bash
c1/.venv/bin/python -m \
  experiments.dense_player_bags.evaluate_cross_feature_holdout \
  --device cpu
```

## Follow-up ablations

All follow-ups below retained the fixed 400-player validation cohort and did
not promote a result to holdout unless stated otherwise.

| Variant | Validation result | Decision |
| --- | ---: | --- |
| Four question-size-specific blend weights | 0.65655 | Rejected: locked internal holdout was 0.664393, below 0.672294. The proxy includes intermediate query sizes, so it is only a diagnostic. |
| Two additional Transformer seeds, averaged with the original | 0.65063 | Rejected. The original seed remains the strongest. |
| DeepSets rank-logit control | 0.64525 | Rejected. |
| 114 raw decision distribution statistics | 0.64779 | Rejected. |
| 192D cached raw move embedding alone | 0.65039 | Rejected as a standalone replacement; retained as a complementary ensemble branch. |

The next high-value feature family is normal-KataGo value analysis: policy
rank, winrate/score loss, and ownership change. Existing value shards cover
only two games per player, so the 36,000 newly selected dense games must be
extracted before this can be tested fairly.
