# Dense value / score-loss features (2026-10-12) - rejected

Same 3,000 / 2,000 pilot games, 96 decisions, single seed. Value tokens = win rate / score lead before and after the move, ownership change
(`extract_value_features.py --samples-per-game 96`, normal model, 1 visit, about 0.73 games/s per worker).

| input | per-game top-1 | best inner NLL | best epoch |
| --- | --- | --- | --- |
| 4-model dense features (production) | 0.3020 | 1.7151 | 28 |
| + value features | 0.3110 | 1.7367 | 14 |

+0.009 per-game (within noise, about +-0.004), NLL worse and earlier overfitting. A full 28,000-game re-extraction would take about 8 h for under +0.01. Rejected.
Compare with `../value_player_bags/` which tests value statistics inside the player-bag models.
Layout: `features/` (value tokens and pilot datasets), `results/`, `logs/`, plus the two build/run scripts.
