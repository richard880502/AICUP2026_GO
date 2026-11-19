# Ten-profile production Transformer — P1 passes; the gain sits away from the hard boundaries

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
