# Ten-profile HumanSL in the production Transformer

Branch: `experiment/ten-profile-production`

## Why this experiment exists

The rankmle premise audit found that, with the same low-capacity learner, ten HumanSL profiles beat the production three-profile set by:

`+0.014674` competition utility, paired player-bootstrap 95% CI `[+0.000680, +0.028837]`.

The gain did not come from rankmle's maximum-likelihood rule or from surprisal alone. It appeared when the learner received all three HumanSL channels (surprisal, log policy rank, policy entropy) under the extra rank profiles.

That audit does NOT show an improvement over the production Transformer. The production model already reads per-decision sequences and may absorb most of the same information. P1 therefore asks the narrow matched question first.

## P1 primary question

> Holding extraction, games, architecture, seeds, epochs, pooling and decoding fixed, does giving the production Transformer all ten HumanSL profiles beat the current three-profile input?

### Arms

| Arm | Streams | Implementation |
| --- | --- | --- |
| `tenp3` | normal + 5k + 1d + 5d | select indices `0,3,5,9` from the ten-profile cache |
| `tenp10` | normal + 11k,8k,5k,2k,1d,2d,3d,4d,5d,6d | use the full cache |

Both arms use `transformer_rank.py` defaults:

- 3 seeds
- 40 epochs
- d_model 128
- 3 Transformer layers
- cumulative-link ordinal head
- NLL training
- player-disjoint inner early stopping
- identical token dropout/noise/optimizer schedule

Both arms are trained from scratch. `--init-from` is deliberately not used because the input dimensionality changes.

## Why the comparison is strongly matched

Only one ten-profile KataGo/HumanSL extraction is performed. The 3-profile baseline is a slice of those exact tensors.

This controls for:

- SGF set
- sampled decision positions
- KataGo/HumanSL evaluation randomness
- masks / rejected positions
- labels
- question samples
- model class and optimization budget

The selected profile indices are checked before training:

`0,3,5,9 -> normal, rank_5k, rank_1d, rank_5d`.

## Primary endpoint

Validation players are sampled into 30 questions each, 5-20 games per question, fixed seed 0.

For each question:

`per-game probabilities -> log-mean pooling -> expected competition-utility decoding`

Primary delta:

`utility(tenp10) - utility(tenp3)`

Player, not question, is the bootstrap unit because the 30 simulated questions of one player overlap.

## P1 advancement gate

Proceed to expensive cross-fitted holdout3 / C1+C2 integration only if:

`delta >= +0.003`

and

`paired player-bootstrap 95% lower bound > 0`.

The report also includes:

- 5-9 / 10-14 / 15-20 game buckets
- per-rank utility
- all nine adjacent-boundary pair-odds AUCs
- changed / improved / worsened decisions
- ten alternative question-sampling seeds

These are secondary diagnostics, not additional tuning targets.

## Run

From repository root:

```bash
git fetch origin
git checkout experiment/ten-profile-production

bash competition2_rank_prediction/experiments/ten_profile_production/run_p1.sh
```

Useful overrides:

```bash
EXTRACT_GPU=1 TRAIN_GPU=1 WORKERS=6 \
OUT=/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production \
bash competition2_rank_prediction/experiments/ten_profile_production/run_p1.sh
```

The extraction is resumable. If the validation ten-profile cache from `rankmle_audit` is complete it is reused.

## Outputs

Default:

`/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production/`

Important files:

- `models/val_probs_tenp3.npz`
- `models/val_probs_tenp10.npz`
- `models/tenp3_seed{0,1,2}.pt`
- `models/tenp10_seed{0,1,2}.pt`
- `p1_report.json`
- `logs/train_tenp3.log`
- `logs/train_tenp10.log`

## P2: holdout3 cross-fit, C1+C2 and v47

P1 passed (`RESULTS.md`), so P2 asks whether the ten-profile gain survives the full pipeline.

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `p2_train.sh` | trains `tenp3_c{0,1}` (`--keep-profiles 0,3,5,9`) and `tenp10_c{0,1}` on `feat10_train` with `--exclude-players features/holdout3/exclude_c{c}.txt`; otherwise the P1 recipe (3 seeds, 40 epochs) |
| 2 | `p2_score.py` | scores each fold's ensemble on `feat10_val` and `feat10_holdout3`; reads `keep_profiles` from the checkpoints and slices the stream axis exactly like `transformer_rank.load_splits`; refuses to write unless it reproduces the training script's `val_probs_<tag>.npz` |
| 3 | `p2_evaluate.py` stage A | replays the locked C1+C2 cross-fit (`crossfit_cohort/evaluate.py`) with `d96` **replaced** (fusion member and C1 residual input); the production replay must give 0.7170117549 |
| 4 | `p2_evaluate.py` stage B | refits the v47 opponent-identity layer (`opponent_identity/h3_final_v47.py`) on the two holdout3 halves with each arm's C1+C2 question probabilities as `base`; the original saved base is refitted in the same run |

Gate (primary, stage B): `v47(tenp10) - v47(tenp3) >= +0.003`, paired player-bootstrap 95% lower bound `> 0`,
and both holdout3 halves `>= 0`. Stage A is reported as secondary.

Run everything from the research root (`runtime/competition2_research`) once the holdout3 ten-profile cache is complete:

```bash
GPU=1 bash experiments/ten_profile_production/run_p2.sh
```

Outputs go to `/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production/p2/` (`models/`, `probabilities/`,
`p2_report.json`, `p2_report.npz`, `logs/`). Results: `RESULTS.md`, section P2.
