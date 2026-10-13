# Dense player-bag status

Current phase: feature extraction complete; dense embedding and bag training
and validation are complete.

- Selected players: 2,000 total, exactly 200 per rank.
- Target games: 20 per player, 40,000 total.
- Existing dense games reused: 4,000.
- Additional games being extracted: 36,000 in 72 shards.
- External validation overlap: 0 players.
- Holdout overlap: 0 players.
- Completed shards: 72/72 (36,000/36,000 games), verified with zero malformed
  shards on 2026-10-05 UTC.
- Eight-worker benchmark: 8 shards completed in about 34 minutes after
  startup, or roughly 118 games/minute.
- Final extraction used 12 KataGo processes across two RTX 3090 GPUs (6 per
  GPU) and completed successfully.
- Both GPUs are currently reserved by the user's separate VLLM service; it is
  not part of this experiment and must not be stopped.

Result:

- Frozen 384D game embeddings were built entirely on CPU without using VLLM
  GPUs: train 40,000 games, validation 8,000, holdout 4,950.
- A cross-game Transformer using only 384D frozen embeddings improved external
  validation from 0.6330 to 0.6486 after validation-selected blending, but its
  locked holdout blend was 0.660312 versus production 0.660856 (delta
  -0.000544). It is rejected.
- Appending 30 frozen per-game d96 ordinal rank-logits (414D total) raised the
  validation blend to 0.654050. With the validation-locked 0.34 bag weight,
  the single holdout evaluation reached 0.672294 versus 0.660856 (delta
  +0.011438; paired bootstrap 95% interval [-0.018847, +0.041321]). This is
  the earlier leading candidate.
- A separate 606D branch appended the cached 192D raw-move representation.
  It was weaker alone, but its errors were complementary. A validation-locked
  0.30 blend of the 414D and 606D bag predictions, followed by a locked 0.28
  blend with production, scored **0.677955** on holdout. This is +0.017098
  over production and +0.005661 over the 414D candidate. The paired 95%
  intervals are [-0.008708, +0.042902] and [-0.009482, +0.021464],
  respectively, so this is the new leading candidate but not a statistically
  confirmed replacement.
- Rejected after validation: size-conditioned blend weights (the only extra
  holdout diagnostic reached 0.664393), a three-seed Transformer average
  (0.650635), a rank-logit DeepSets control (0.645249), 114 raw decision
  statistics (0.647793). The 192D frozen raw-move branch was also weaker by
  itself (0.650391) but was retained only as a complementary ensemble branch.
  These figures are validation fusion scores unless stated otherwise.

Prepared next stages:

1. `extract_game_embeddings.py` concatenates three frozen `d96_h` 128D game
   representations into a 384D vector for train/validation/holdout.
2. `attach_game_logits.py` appends 30 frozen ordinal rank-logits without
   invoking KataGo again.
3. `train_bags.py` trains a direct 5-20 game Set Transformer with 300
   player-disjoint inner-validation players.
4. Only if external validation improves will a blend be locked and evaluated
   on holdout.
5. The remaining high-upside feature family is normal-KataGo value/ownership
   analysis. It requires extracting the 36,000 extra player-bag games; do not
   mix the existing two-game-per-player value shards into 20-game bags.

Reproduce the locked current candidate without holdout tuning:

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m \
  experiments.dense_player_bags.evaluate_cross_feature_holdout \
  --device cpu
```

## Decoder constraint diagnostic

A validation-only test of a global Hungarian decoder was run on ten separate
400-player cohorts with exactly 40 players per rank. Although the internal
holdout had suggested that a uniform rank-count constraint might help, it did
not transfer to the cross-feature candidate: the constrained decoder changed
the 5/10/15/20-game scores from 0.6069/0.6621/0.6774/0.6920 to
0.6114/0.6591/0.6738/0.6823. It is rejected, unless the organizer explicitly
guarantees exact per-rank test counts and a fresh validation protocol supports
it. The reproducible validation-only report is at
`results/global_decode_validation/report.json`.

## Active high-signal experiment

`../value_player_bags/` is extracting 32 normal-KataGo value and ownership
features for all 40,000 selected games. This fills the coverage gap that made
the earlier value experiment unfair: its value features covered only about two
games per player, while the strongest bag models use 20. The new branch will
append robust per-game summaries of value loss and ownership change to the
current 414D / 606D representations, then be selected solely on validation.

The extraction command is resumable:

```bash
cd /tmp2/richard/rank_prediction
bash experiments/dense_player_bags/run_extraction.sh
```
