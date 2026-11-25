# Ten-profile production Transformer — P1 passes, P2 fails: the gain does not survive the pipeline

**Overall decision (2026-11-19): do not adopt ten profiles in production.** P1 below shows a real d96-level
gain; P2 at the end shows it vanishes inside C1+C2 and is within pipeline noise on top of v47, where both matched
arms score below the current production model.

Completed 2026-11-19. Validation cohort (400 players, 40 per rank), d96 production Transformer alone. Exploratory: not a
holdout3, C1+C2, v47, Public or submission result.

## Decision

**P1 passes, with margin.** Same ten-profile extraction, same `transformer_rank.py` defaults and budget, 3 seeds per arm;
the only difference is that `tenp3` keeps tensor indices `0,3,5,9` (normal, rank_5k, rank_1d, rank_5d):

| Arm | Utility | Exact | Off by one | Off by two or more |
| --- | ---: | ---: | ---: | ---: |
| `tenp3` | 0.5928 | 0.4390 | 0.4182 | 0.1428 |
| `tenp10` | **0.6067** | **0.4565** | 0.4083 | 0.1352 |

```text
tenp10 - tenp3   +0.0139   paired player-bootstrap 95% CI [+0.0072, +0.0208]
gate (>= +0.003 and lower bound > 0)   PASS
```

P2 is justified. It is not yet a production result.

## Robustness

| Check | Result |
| --- | --- |
| 10 question-sampling seeds | **10/10 positive**, mean **+0.0094**, min +0.0053, max +0.0139 |
| decisions changed | 1,249 of 12,000: 738 improved, 434 worsened, 10,828 unchanged |
| game-count buckets | +0.0130 (5-9), +0.0139 (10-14), +0.0146 (15-20) |
| inside training (inner slice, seed 0) | best NLL 1.6910 -> 1.6791; per-game top-1 0.3071 -> 0.3164 |

The preregistered primary seed (0) is the highest of the ten sampling seeds. It was fixed before the run, so the gate
reading stands, but **+0.009 is the fairer central estimate**. The sampling seeds reuse the same trained models and the
same 400 players.

This survives the production learner at about the size the fixed-learner audit found (`rankmle_audit`, mean +0.0117 over
20 seeds). The expectation that the Transformer had already extracted this from three profiles was wrong.

## Where the gain is — and where it is not

| Rank | D | C | B | A | 1D | **2D** | 3D | 4D | 5D | 6D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tenp3` | 0.8103 | 0.5950 | 0.6218 | 0.7504 | 0.4434 | 0.3524 | 0.4553 | 0.4583 | 0.6269 | 0.8145 |
| `tenp10` | 0.8476 | 0.5901 | 0.6543 | 0.7459 | 0.4498 | 0.3522 | 0.4759 | 0.4773 | 0.6348 | 0.8393 |
| change | **+0.037** | -0.005 | **+0.033** | -0.005 | +0.006 | **0.000** | +0.021 | +0.019 | +0.008 | **+0.025** |

| Boundary AUC | D/C | C/B | B/A | A/1D | **1D/2D** | 2D/3D | 3D/4D | 4D/5D | 5D/6D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `tenp3` | 0.8044 | 0.8168 | 0.8476 | 0.6035 | 0.7322 | 0.7826 | 0.8218 | 0.7199 | 0.8425 |
| `tenp10` | 0.8128 | 0.8279 | 0.8494 | 0.6176 | **0.7279** | 0.7890 | 0.8240 | 0.7293 | 0.8544 |

The gain concentrates at the extreme ranks (D, B, 6D). **The hardest boundary, 1D/2D, gets slightly worse and 2D
utility does not move.** That is consistent with everything else recorded on this problem: finer playing-strength
resolution helps where strength separates registered ranks and does nothing where registered rank and strength
disagree.

## The risk for P2

v47 already resolves the extremes well through opponent identity (holdout3: D/C 0.908, 5D/6D 0.906, against 1D/2D
0.777; see `hard_boundary_atlas`). The ten-profile gain is concentrated exactly where v47 is strong and absent where
it is weak. Earlier own-move improvements collapsed once combined with the rest of the pipeline (`full_train_coverage`
-0.00058; `strength_score_v2` zero fusion weight). P2 should therefore expect substantial shrinkage after the C1+C2
replay and the v47 fusion, and keep the same strict gate.

## How this was run

The training cache was extracted on a shared host whose memory was exhausted by another user's job (about 127 GB
resident plus swap). The first attempt with the stock config — six workers at about 8.3 GB of KataGo each — pushed the
host into swap and was stopped before writing any shard.

The completed run used `lowmem_p1.sh` and a low-memory copy of the analysis config:

| Setting | Stock | Low-memory |
| --- | ---: | ---: |
| `numAnalysisThreads` | 512 | 64 |
| `nnMaxBatchSize` | 512 | 64 |
| `numNNServerThreadsPerModel` | 2 | 1 |
| measured per worker (KataGo + Python, steady) | ~8.3 GB | ~6.2 GB |

These change concurrency only, not the 1-visit raw-policy outputs beyond FP16 batching noise; symmetry is not fixed in
either extraction, so both caches already carry symmetry noise. The validation cache (rankmle audit) used the stock
config and the training cache the low-memory one. **Both arms read both caches, so the comparison stays matched.**

`lowmem_p1.sh` starts workers only after MemAvailable stays above a threshold for 90 seconds, sizes the worker count
from free memory, stops every worker if MemAvailable falls below 1.5 GB (with a 10-minute cooldown), and then runs the
unchanged `run_p1.sh`. It tracks processes by PID only. A 40-game smoke test measured 3.7 GB per worker, which
under-states the steady 6.2 GB; the driver was restarted once with the measured value so its restart arithmetic would
not oscillate. Timeline: extraction 11:23-12:59 with four workers, training and evaluation 12:59-13:04.

Paths are for the w77 research host: run from `runtime/competition2_research`, outputs under
`/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production/`.

## P2 — holdout3 cross-fit: the gate fails

Completed 2026-11-19. 2,000 holdout3 questions, cross-fitted over the two recorded halves. Per fold, `tenp3_c{f}` and
`tenp10_c{f}` were trained on the ten-profile training cache with that half's holdout3 players excluded (3 seeds, 40
epochs, P1 defaults), scored on the validation and ten-profile holdout3 caches, and swapped in for the `d96` member of
the locked C1+C2 recipe (`res`, `oppC2` and the C1 residual unchanged). Stage B rebuilds the v47 cross-fit with each
arm's C1+C2 question probabilities as `base`.

### Sanity replays — all exact

| Check | Result |
| --- | --- |
| scorer reproduces each arm's training-time validation probabilities | max abs diff 0.0, argmax agreement 1.0 |
| stage A with the production d96 | 0.7170117549 = recorded 0.7170117549 |
| stage A production question probabilities vs `full_qprob.npz` | max abs diff 0.0 |
| stage B, original base and production replay | both 0.765302 |

### Results

| Stage | Production | `tenp3` | `tenp10` | `tenp10 - tenp3` | 95% CI | Halves |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A — C1+C2 | 0.7170 | 0.7162 | 0.7157 | **-0.0005** | [-0.0046, +0.0037] | -0.0034 / +0.0025 |
| **B — v47 (primary)** | **0.7653** | 0.7614 | 0.7636 | **+0.0022** | **[-0.0006, +0.0052]** | +0.0013 / +0.0032 |

Gate (`>= +0.003`, lower bound `> 0`, both halves `>= 0`): **fails** on the first two criteria.

Refit sensitivity (base jittered by `exp(1e-6 N(0,1))`, five refits, jitter shared across arms): `tenp10 - tenp3` =
+0.0022, +0.0014, +0.0035, +0.0030, +0.0020 — mean **+0.0024**, SD 0.0008. Each arm's own refit SD is about 0.00045.
The direction is consistent; the size stays under the gate.

### Stage A: the d96 gain disappears inside C1+C2

On validation the fold models reproduce P1's d96-level gain (`transformer_rank.py`'s own report: +0.0096 in both
folds). Inside C1+C2 it is gone: only **55 of 2,000** holdout3 decisions change between the arms (27 better, 26 worse),
and C1+C2 1D/2D AUC is 0.7257 (production), 0.7243 (`tenp3`), 0.7270 (`tenp10`). The fusion weights barely move
(`d96` 0.70-0.73, `res` 0.31-0.33, `oppC2` 0.54-0.55). The ResNet member, the opponent-side member and the C1
residual already carry what the extra profiles add to d96.

### Stage B: within noise, and below production

`tenp10 - tenp3` changes 26 decisions (16 better, 9 worse); boundary AUCs move by less than 0.001 (1D/2D 0.7776 ->
0.7782). The more important number is the matched control against production:

| Pair, stage B | Delta | 95% CI |
| --- | ---: | --- |
| `tenp3 - production` | **-0.0039** | **[-0.0075, -0.0006]** |
| `tenp10 - production` | -0.0017 | [-0.0054, +0.0019] |

`tenp3` carries the same three profiles and the same games as production's d96, yet re-extracting and retraining it
costs 0.0039 at the v47 level with an interval that excludes zero. **A single re-extract-and-retrain moves v47 by
more than the ten-profile benefit.** Replacing production with `tenp10` would lose about 0.0017.

Sources of that replication gap cannot be separated here: training-seed variance, FP16 batching differences from the
low-memory extraction config, and KataGo's random symmetry in both extractions all contribute. It is also a caution
for every earlier single-retrain comparison on this cohort: differences under about 0.004 at the v47 level are not
reliably distinguishable from re-running the pipeline.

### Reading

This matches the risk stated before P2. The ten-profile gain is real for d96 alone, concentrated at the extreme ranks
where v47 is already strong, and absent at 1D/2D where v47 is weak. Fused with the rest of the pipeline it no longer
moves decisions. Do not extract ten profiles for the full corpus or the official test set.

### Fixes during P2

- `p2_score.py` crashed on holdout3: its shards store an empty rank for every game (labels live at question level in
  `question_folds.csv`), so the per-game top-1 diagnostic could not map them. It is now `None` for unlabelled splits;
  the scored probabilities were unaffected.
- The first stage-B dry run fed arm probabilities as float64 while production's d96 is float32; stage A matched but v47
  moved 0.765302 -> 0.764486. Arms are now read as float32, and every replay matches exactly.
- The holdout3 cache was extracted with `lowmem_extract.sh` (the extraction-only form of `lowmem_p1.sh`). One run was
  stopped at MemAvailable 1.36 GB while extra P2 work shared the host, and resumed with three workers; 25 shards,
  24,822 games, cache check passes.
