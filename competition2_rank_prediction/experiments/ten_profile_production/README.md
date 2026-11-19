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

## Conditional P2

P2 is intentionally not implemented before P1 passes.

If P1 passes, the next experiment will:

1. extract ten-profile own-side features for holdout3;
2. train two ten-profile cross-fit models excluding holdout3 half 0/1 exactly like `d96_c0/c1`;
3. replace only the d96 member inside the frozen C1+C2 replay, refitting fold-safe fusion/calibration;
4. compare matched `C1+C2(3-profile)` vs `C1+C2(10-profile)`;
5. only if that passes, combine with frozen v47 opponent-identity evidence.

The eventual decision metric is not the simple learner's +0.0147. It is the incremental utility over the current full pipeline.
