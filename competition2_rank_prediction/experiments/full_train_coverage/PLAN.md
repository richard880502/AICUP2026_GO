# Full-player training coverage experiment

## Hypothesis

The current dense rank encoder was trained on about two games per training
player. The source contains many more games for the same players. Adding six
different games per existing training player may improve the per-game rank
signal and its contribution to the current C1+C2 ensemble.

## Fixed protocol

- Include every player in the existing dense training split.
- Sample up to six additional games per player with one fixed seed. Exclude
  all already extracted games and all external-validation players. If a
  training player has fewer than six unseen games, use every available game
  and record the per-player coverage.
- Keep the existing validation and cross-fitted 2,000-question evaluation
  cohorts unchanged.
- Train the same three-seed d96 Transformer with the same settings, using the
  original and added feature shards together. Early stopping remains grouped
  by player.
- Fit any ensemble weights and C1 calibration on the existing validation
  cohort only. Evaluate the resulting candidate once on the cross-fitted
  cohort.

## Advancement gate

Advance only if the full C1+C2 candidate improves utility by at least 0.003,
the paired 95% player bootstrap interval is above zero, and both cross-fitted
halves are non-negative. Otherwise keep the current production model.

This experiment uses one physical GPU (GPU 1) for KataGo extraction. It does
not modify official submissions or the GPU 0 vLLM service.

## Prepared input

`prepare.py` wrote 45,013 additional games from the 7,607-player training
split. It chose six games for 7,348 players and all remaining games for 259
players with fewer than six unused records. Ten players had no unused source
games. There are zero game-key duplicates against the existing training
features and zero validation-player overlap. The selection seed is 20261109.

Extraction is resumable:

```bash
bash competition2_rank_prediction/experiments/full_train_coverage/run_extraction.sh
```

The launcher defaults to six workers and physical GPU 1. It has not been
started while both GPUs are occupied by separate services.

## CPU fallback benchmark, 2026-11-09

To avoid using either occupied GPU, I built KataGo 1.15.0 with Eigen, AVX2,
and FMA in `/tmp2/richard/build/katago_eigen`. A CPU benchmark with the
authorized normal network measured 6.36 neural evaluations per second using
four Eigen threads. The extraction needs roughly 90 decisions per game and
four policy profiles, which implies about 30 days on one such engine for the
45,013-game batch; parallel CPU workers would still take multiple days. The
build is retained outside the repository for small CPU-only checks; the full
extraction still requires GPU capacity.

After extraction, train the same three-seed Transformer separately for each
cross-fit half. The launcher excludes that half's 1,000 players before
training, scores validation and holdout3, and does not overwrite the original
checkpoints:

```bash
bash competition2_rank_prediction/experiments/full_train_coverage/run_training.sh
```
