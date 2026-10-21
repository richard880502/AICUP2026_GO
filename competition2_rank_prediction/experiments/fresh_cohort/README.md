# Fresh player-disjoint cohort (2026-10-20)

Purpose: confirm the opponent-side member (`experiments/opponent_side`) on players that no model and no selection step has seen.
Validation gave +0.053 [+0.040, +0.065] (400 players) but the old holdout only +0.020 [-0.003, +0.043] (400 players), so a third,
untouched cohort decides.

- Cohort: `scripts/make_holdout_test.py --seed 20261020 --out features/holdout2 --exclude features/holdout/players.txt`
  - 400 new players (40 per rank, >= 20 games, not in validation or the old holdout), one 5-20 game question each, 5,076 games.
- Models `d96_h2` and `resnet8c_h2` (in `results/`) are retrained excluding the old holdout AND this cohort
  (`features/holdout2/exclude_h2.txt`), with the production recipes.
- Features: `features/feat96/holdout2` (player) and `experiments/opponent_side/features/feat96opp/holdout2` (opponent).
- `run.sh` launches training and extraction; `finish.py` makes the snapshot-mean validation probabilities of the ResNet, runs
  `predict_test.py --suffix _h2 --opp-feat ... --dump`, and scores the comparison fixed in advance:
  - best2 vs best2 + opponent;
  - d96 vs d96 + opponent.
- Weights and offsets are fitted on validation only.

## Result (2026-10-20 23:10 Taipei)
| method | fresh cohort (400 new players) |
| --- | --- |
| best2 (d96_h2 + resnet8c_h2) | 0.6535 +-0.018 |
| **best2 + opponent** | **0.7206 +-0.017** |
| d96_h2 alone | 0.6381 |
| d96_h2 + opponent | 0.7147 |
| opponent alone | 0.7016 |

best2 + opponent minus best2: **+0.067 [+0.046, +0.090]**; d96 + opponent minus d96: +0.077 [+0.051, +0.101].
Confirmed on players no model or selection step had seen. Across the three player-disjoint cohorts the gain is
+0.053 (validation), +0.020 (old holdout), +0.067 (fresh) - always positive, about +0.05 on average.

## Confusion matrices (`confusion.py` -> `figures/confusion_best2_vs_opponent.png`, `results/confusion_summary.json`)
| cohort / method | exact | off by one | off by >= 2 |
| --- | --- | --- | --- |
| fresh: best2 | 0.510 | 0.390 | 0.100 |
| fresh: best2 + opponent | 0.590 | 0.355 | 0.055 |
| old holdout: best2 | 0.527 | 0.362 | 0.110 |
| old holdout: best2 + opponent | 0.530 | 0.410 | 0.060 |
On both cohorts the large errors (>= 2 ranks) roughly halve; on the fresh cohort exact hits also rise at almost every rank
(B 0.60 -> 0.75, 1D 0.38 -> 0.50, 3D 0.40 -> 0.53, 4D 0.33 -> 0.45). A remains the weakest row (0.38).
