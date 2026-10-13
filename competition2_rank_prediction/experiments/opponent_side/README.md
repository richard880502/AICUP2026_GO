# Opponent-side decisions (2026-10-12) - rejected

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
