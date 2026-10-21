# Opponent-side decisions (2026-10-12) - INVALID, being re-run

**2026-10-20: the result below is void.** `extract_rank_features.py --opponent` parsed the flag but never used it, so
`feat96opp` contained the labelled player's own moves again (35% of values bit-identical, per-game mean surprisal corr 0.9995;
the rest differ only by KataGo's random symmetry). Fixed (the flag now swaps the sampled colour); the buggy features are kept in
`features/feat96opp_BUGGY_same_player`, `features/pilot_opp` was built from them and must be rebuilt.


Hypothesis: opponents are paired at a similar strength, so their moves are extra evidence about the same game.
Pilot on the same 3,000 train / 2,000 validation games, 96 decisions, single seed. `extract_rank_features.py --opponent` extracts the
opponent's side; `build_opp_pilot.py` builds player-only / opponent-only / both datasets; `run_opp_pilot.sh` runs the extraction.

| input | per-game top-1 | best inner NLL |
| --- | --- | --- |
| player side only (production) | 0.3020 | 1.7151 |
| opponent side only | 0.2935 | 1.7276 |
| player + opponent (192 decisions) | 0.3040 | 1.7307 |

The opponent carries almost as much rank information as the player, but the two are redundant: combining gives no gain. Rejected.
Layout: `features/` (opponent features and pilot datasets), `results/`, `logs/`. Question-level scoring errors in the logs are expected (validation players had < 5 games in the 2,000-game subset).

## v2 pilot with the corrected features (2026-10-20)
Same 3,000 train / 2,000 validation games, now 3 seeds per variant (`results_v2/`, `eval_v2.py`). Per-game, paired player bootstrap
vs player-only:

| input | per-game score | top-1 | NLL | score diff [95%] |
| --- | --- | --- | --- | --- |
| player only | 0.4576 | 0.309 | 1.712 | - |
| opponent only | 0.4163 | 0.280 | 1.791 | -0.041 [-0.065, -0.018] |
| player + opponent (192 decisions, one model) | 0.4944 | 0.328 | 1.610 | +0.037 [+0.019, +0.055] |
| late fusion of separate player / opponent models | 0.4939 | 0.337 | 1.653 | +0.037 [+0.021, +0.054] |
| late fusion, PLAYER model applied to opponent moves | 0.4881 | 0.327 | 1.649 | +0.031 [+0.013, +0.049] |

Opponents are matched at nearly the same strength (`experiments/a1d_boundary`: per-game corr 0.80, rank medians equal), so the
opponent's moves are a second, partly independent sample of the same rank. The last row needs no new model: the production d96_h
can be applied to opponent features directly. Next: `run_full_opp.sh` (opponent features for all 8,000 validation games and the
holdout; 1 KataGo per GPU - 2 per GPU runs out of memory next to the other users' jobs) then `eval_holdout.py`
(best2 vs best2 + opponent member, weights/offsets fitted on validation only). Note this is a per-game pilot; the question-level
gain on the holdout can be smaller because 5-20 games already average out per-game noise.

## Full validation + holdout (2026-10-20, `run_full_opp.sh`, `eval_holdout.py`, `eval_val_questions.py`)
Opponent features now exist for all 8,000 validation games and the 4,950 holdout games (`features/feat96opp/{val,holdout}`).
The production d96_h models are applied unchanged to the opponent's decisions (member `d96_h_opp`).

| method | holdout (400 q) | validation (8,000 q, 5-fold CV fit) |
| --- | --- | --- |
| best2 | 0.6609 | 0.6350 |
| best2 + opponent | 0.6808 | 0.6875 |
| d96_h + opponent | 0.6837 | 0.6858 |
| d96_h alone | 0.6441 | 0.6249 |
| opponent alone | 0.6738 | 0.6834 |

best2 + opponent minus best2: holdout +0.020 [-0.003, +0.043]; validation +0.053 [+0.040, +0.065].
Reproduced through `predict_test.py --suffix _h --opp-feat experiments/opponent_side/features/feat96opp/holdout` (0.6808).
The opponent member alone beats the player's own moves at question level: opponents are matched by the label rank and differ from
game to game, so their average tracks the label, while the player's own moves keep a player-specific offset.
