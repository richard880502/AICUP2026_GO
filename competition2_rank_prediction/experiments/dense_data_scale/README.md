# Data-scale curve for the dense Transformer (2026-10-06) - more data does not help

Single seed, validation cohort, `transformer_rank.py --train-fraction`:

| training games | per-game top-1 | question (log-mean + expected) |
| --- | --- | --- |
| 3,802 (25%) | 0.2946 | 0.588 |
| 7,604 (50%) | 0.3019 | 0.590 |
| 15,211 (100%) | 0.3056 (3 seeds 0.305-0.310) | 0.591 (3-seed ensemble) |

Each doubling gives +0.007 then +0.004 per game: data volume is not the bottleneck.
