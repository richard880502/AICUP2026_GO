# Competition 2 (rank prediction) - folder index

Working directory: `/tmp2/richard/rank_prediction/`. Competition 1 is reused read-only through the `c1` symlink; the official training data is `data/`.

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
- `features/`: `feat96/{train,val,holdout,holdout2}` dense features (production), `feat96opp/{val,holdout,holdout2}` opponent-side features (production since 2026-10-20), `holdout2` (fresh 400-player cohort), `feat`, `feat_value` (32-decision legacy), `moves`, `shards`, `holdout` (400-player untouched test + labels), `rehearsal`, `move_embeddings_h`.
- `results/`: only what `predict_test.py` needs (see `results/README.md`). `archive/`: recoverable superseded checkpoints, ablations and early scratch (nothing deleted).
- `experiments/<name>/`: one directory per experiment with its own README, `results/` and `logs/`. Accepted/rejected status is listed in `CURRENT_BEST.md`; the 2026-10-11/12 additions are
  `loss_and_head`, `opponent_side`, `value_pilot_dense`, `game_level_features`, `dense_data_scale` (all rejected).
- `logs/`: training and extraction logs of the production pipeline.

## Experiments after the C1+C2 freeze (2026-11-06 to 11-20)

None of these changed a submission; the final rank submission is v47
(Public 0.7599623, `experiments/opponent_identity/`). Each directory has its own
README / RESULTS. Their `production/` artifacts (checkpoints, predictions,
feature caches) were deleted on 2026-11-25; the list of deleted paths is kept on
the research machine in `production/research_scripts_20261116/cleanup_20261125_deleted.txt`.

| Experiment | Date | Outcome |
| --- | --- | --- |
| `adjacent_rank_specialist` | 11-07 | Player-level probe on frozen embeddings for adjacent ranks (issue #1 pilot); not adopted |
| `semi_hard_mining` | 11-08 | Ordinal semi-hard pair mining, three rounds; all gates fail (issue #1) |
| `mining_submission` | 11-08 | Joint alternate submission (`docs/submission_joint_20261108.md`): Public 0.7195608 vs 0.7227214; not adopted |
| `reward_alignment` | 11-08 | Expected-score aligned retraining of d96; paired interval crosses zero |
| `reward_calibration` | 11-09 | +0.00008, interval crosses zero; gate fails |
| `decision_alignment` | 11-09 | +0.0002, interval crosses zero; gate fails |
| `dense_data_scale` (coverage / fusion / phase notes) | 11-09 | The cache holds at most two games per player, so the old curve does not show saturation; follow-ups negative |
| `question_supervision` | 11-09 | Question-level training signal into the token encoder; not adopted |
| `search_regret` | 11-09 | MCTS-revealed information pilot; not adopted |
| `human_finetune` | 11-09 | HumanSL fine-tuning: +0.008 on the reused 400-player validation cohort (exploratory, no fresh confirmation); not adopted |
| `policy_adaptation` | 11-09 | Stopped at the inner gate (+0.001, required +0.01) |
| `human_trunk` | 11-09 | HumanSL trunk probes (linear and temporal) fail the utility gate |
| `human_context` | 11-09 | Human-profile value and uncertainty features; not adopted |
| `seed_ensemble` | 11-09 | More seeds of C1+C2 on holdout3; not adopted |
| `query_meta` | 11-09 | Residual MLP replacing the question-level calibration; not adopted |
| `label_shift` | 11-09 | Label-shift correction on the borrowed-opponent branch; not adopted |
| `hd1_komi` | 11-09 | HD1 komi hypothesis: +0.0005, interval crosses zero |
| `training_overlap` | 11-09 | Zero move-sequence matches between the 1M training games and the 800 official questions |
| `dense_encoder_scale` | 11-10 | Expanded d96 training: -0.0021 vs C1+C2; both folds regress |
| `full_train_coverage` | 11-17 | Full training coverage extraction; negative |
| `strength_score_v2` | 11-17 | Strength Score v2 and Tier A (`opponent_identity/TIER_A_RESULTS.md`); all negative |
| `rank_context` | 11-18 | The handicap channel is empty in the official test set |
| `opponent_identity` audits | 11-18 | Shared unknown opponents (`SHARED_OPPONENT_AUDIT.md`) and identity / matchmaking graph (`IDENTITY_MATCHMAKING_AUDIT.md`); both negative |
| `hard_boundary_atlas` | 11-18 | Nothing usable at 1D/2D against a permutation null |
| `rankmle_audit` | 11-19 | Ten HumanSL profiles vs three: +0.0147 for a fixed learner, underpowered |
| `ten_profile_production` | 11-19 | +0.0139 in the production Transformer, but the gain does not survive C1+C2 or v47 (issue #7) |
| `conditional_complementarity` | 11-20 | Premise does not hold |
| `opponent_rank_posterior_audit` | 11-20 | Known/unknown control audit; the v47 proxy is not deployable |

The 2026-11-17 to 11-20 entries were first committed on separate branches; those
branches are kept as `archive/*` tags (see the repository README).

## Test-day checklist (details: METHODS.md section 6)
1. `scripts/test_to_games.py --tests-dir <dir>`; `scripts/run_test_extraction.sh` (player AND opponent dense extraction ->
   `features/feat96/test`, `features/feat96opp/test`, then final-position features -> `features/final_result/test`;
   4 KataGo on 2 GPUs, about 3-4 h; `PER_GPU=1` if other jobs fill the GPUs);
   `scripts/prep_test_moves.py`.
2. `predict_test.py --tests-dir <dir>` -> `submission_rank.csv` (800 rows). Default since 2026-10-21 = best2 + opponent-specific
   member (opp_scratch) + shift residual (residual_shift.py); it refuses to run without the opponent / final-result features.
3. First check whether test players overlap training players and whether a player appears in several questions.

## Caveat
All scores are internal (validation cohort and the 400-player holdout), not the official leaderboard. The holdout has now been used for method comparison, so a fresh player-disjoint cohort is needed to confirm the player-bag candidates.
