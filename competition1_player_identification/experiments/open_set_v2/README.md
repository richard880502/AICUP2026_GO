# Open-set v2 player matching

Status, 2026-11-11: protocol manifests prepared; no encoder has been trained
and no new score is claimed. The current official Public remains **0.5236602**.

The training index contains 1,000,000 rows and 8,007 players. Of these, 6,651
have at least 15 games and at least five games in each color. A fixed seed
(20261111) reserves 600 eligible players for model selection and another 600
for a final report, leaving 6,807 training players. Official test and candidate
files do not enter this split.

Each evaluation cohort has 600 candidates, ten reference games per candidate,
and 2,400 questions, balanced across query counts 1/2/3/5. Query games are
disjoint from reference row pointers. Query subsets are nested across counts;
confidence intervals must therefore bootstrap players, not individual questions.
Both colors are allowed without forcing an artificial 50:50 query ratio.

Generate the immutable manifests from the repository root:

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition1_player_identification.experiments.open_set_v2.prepare
```

Output: `production/open_set_v2_20261111/manifests/`. An existing directory is
never overwritten. `summary.json` contains manifest hashes and audit counts.

## Required gates before fitting

- Read indexed CSV rows and verify player/color/path consistency.
- Hash the actual SGF move sequences; remove query/reference duplicates and
  games shared between train, validation and final holdout. Player separation
  alone cannot guarantee game separation: the same game may appear for both
  opponents. The current manifests explicitly mark this audit as pending.
- Audit the actual SGF metadata, game length and color distribution against
  the published v2 rules. The eligible-player selection must be reported as
  a cohort restriction; this proxy is not guaranteed to match leaderboard data.
- Fit all preprocessing and feature reliability weights on training players.
  Select checkpoints and blends on validation only. Keep final-holdout
  outcomes unopened until a fixed configuration is selected.

## Experiment sequence

1. Establish a frozen rich-profile and opening-sequence baseline on the new
   validation protocol. Report exponential Top-5 score, Top1/5/20 and each
   query-count/color subgroup, with player-bootstrap uncertainty.
2. Train a new open-set encoder on the 6,807 other players using episodic
   ten-reference and 1/2/3/5-query matching with both colors. Keep existing
   production encoders frozen as controls. Existing HumanSL caches cover
   400 players / 40,000 reference games, not all 6,807 players; cache expansion
   is necessary, and its coverage must be checked before launching training.
3. Add opening prefix / sequence matching and a raw both-player move encoder.
   Preserve perspective and color; fit fusion only on validation.
4. Compare centroid similarity against game-to-game similarity aggregated over
   ten references, then learned pair scoring. Start with mean/top-k/log-mean-exp
   aggregation, and use validation to choose rather than assume an improvement.
5. Add the Competition 2 rank model as a soft prior. Calibrate rank uncertainty
   and measure true-player recall before testing any hard candidate pruning.

Only the best fixed configuration is evaluated on final holdout. Actual
leaderboard submissions are reported separately from this training-only proxy.

## Checks

`pylint competition1_player_identification/experiments/open_set_v2` scores
10/10; `mypy` on the same target passes. The repository pre-commit hook runs
both tools. The Redmine infra source document is unavailable in this session;
existing local hooks are used, without claiming verification against that text.
