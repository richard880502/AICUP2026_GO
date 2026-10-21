# Tactical "fighting context" features (2026-10-19/20) - not adopted (gain not confirmed on the holdout)

`fight_features.py` (project root) simulates the board (board_sim.py, no KataGo) for exactly the same 96 decisions as the dense features
and records 10 numbers per decision: adjacent opponent / own stones, stones captured, opponent groups put in atari, liberties of the
played group, self-atari, distance to the opponent's previous move, own groups in atari before the move, stone density within distance 2,
line. Output: `features/fight96/{train,val,holdout}` (all 28k games in about 1 minute). Appended with `transformer_rank.py --value-train/--value-val`.

| | baseline dense Transformer | + fight features |
| --- | --- | --- |
| validation, 3-seed per-game top-1 | 0.3115 | 0.3156 |
| validation, question log-mean + expected | 0.5906 | **0.6145** (+0.024) |
| validation, holdout-excluded training (`_h`) | 0.5860 | 0.6110 (+0.025) |
| **holdout, Transformer alone** | 0.6441 | 0.6472 (+0.003, 95% [-0.020, +0.026]) |
| **holdout, with cosine ResNet (production best2)** | 0.6609 | 0.6635 (+0.0026, 95% [-0.014, +0.020]) |

The validation gain was seen in two independent trainings but did not replicate on the untouched holdout. Either the validation cohort
favours these features or the true gain is about +0.01 and hidden by holdout noise; not enough evidence to change production.
Reproduce: `c1/.venv/bin/python experiments/fight_features/eval_holdout.py`. Artifacts in `results/`, logs in `logs/`.
