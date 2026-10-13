# Competition 2 (rank prediction) - folder index

Repository directory: `competition2_rank_prediction/`. On the research
machine, competition 1 is reused read-only through a local `c1` symlink and
the official training data is mounted at `data/`; neither is tracked by Git.

## Which document to read
| file | what it is | authoritative for |
| --- | --- | --- |
| `CURRENT_BEST.md` | latest production pipeline, verified scores, status of every `experiments/*` | **current best and default model set** |
| `METHODS.md` | Traditional-Chinese record of the method, results and pitfalls up to 2026-10-12 | how things work, negative results, test-day steps (sections 6, 11, 12) |
| `README.md` | English command cheat-sheet and early result tables | quick commands (some tables predate CURRENT_BEST.md) |
| `PLAN.md` | the first plan (2026-10-03) | history only (superseded) |

## Code (top level)
- Feature extraction: `extract_rank_features.py` (dense KataGo/HumanSL; `--opponent`, `--profiles`), `extract_value_features.py`, `scripts/prep_moves.py`, `scripts/prep_test_moves.py`, `scripts/run_extraction96.sh`.
- Models: `transformer_rank.py` (ordinal Transformer; `--head --loss --select --game-feats --keep-profiles --token-stride --exclude-players`), `move_transformer.py`, `resnet_rank.py`, `board_sim.py`.
- Evaluation and fusion: `blend_eval.py`, `question_aggregator.py` (optional, off by default), `eval_holdout.py`, `eval_final.py`, `eval_dense_*.py`, `analysis_*.py`, `snapshot_probs.py`.
- Inference: `predict_test.py` (`--model-set best2|full4`, `--suffix _h` for holdout-excluded models, `--balanced`/`--use-aggregator` off by default).

## Data and artifacts
- `features/`: `feat96/{train,val,holdout}` dense features (production), `feat`, `feat_value` (32-decision legacy), `moves`, `shards`, `holdout` (400-player untouched test + labels), `rehearsal`, `move_embeddings_h`.
- `results/`: only what `predict_test.py` needs (see `results/README.md`). `archive/`: recoverable superseded checkpoints, ablations and early scratch (nothing deleted).
- `experiments/<name>/`: one directory per experiment with its own README, `results/` and `logs/`. Accepted/rejected status is listed in `CURRENT_BEST.md`; the 2026-10-11/12 additions are
  `loss_and_head`, `opponent_side`, `value_pilot_dense`, `game_level_features`, `dense_data_scale` (all rejected).
- `logs/`: training and extraction logs of the production pipeline.

## Test-day checklist (details: METHODS.md section 6)
1. `scripts/test_to_games.py`; dense extraction on `features/shards/test` (96 decisions, rank_5k/1d/5d, 4 workers on 2 GPUs, about 1.5-2 h); `scripts/prep_test_moves.py`.
2. `predict_test.py --tests-dir <dir>` -> `submission_rank.csv` (800 rows).
3. First check whether test players overlap training players and whether a player appears in several questions.

## Caveat
All scores are internal (validation cohort and the 400-player holdout), not the official leaderboard. The holdout has now been used for method comparison, so a fresh player-disjoint cohort is needed to confirm the player-bag candidates.
