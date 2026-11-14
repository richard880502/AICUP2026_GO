# Data-scale curve for the dense Transformer (2026-10-06)

## Scope correction (2026-11-09)

This historical curve does **not** establish data saturation. The complete
cached d96 training set contains 15,211 games from 7,607 players, with at most
two games per player (verified directly from every training feature shard).
The curve changes subsamples of this small cache, not coverage toward the
one-million-game source dataset. It does not test additional games from the
same training players. The original numbers below remain unchanged.

Single seed, validation cohort, `transformer_rank.py --train-fraction`:

| training games | per-game top-1 | question (log-mean + expected) |
| --- | --- | --- |
| 3,802 (25%) | 0.2946 | 0.588 |
| 7,604 (50%) | 0.3019 | 0.590 |
| 15,211 (100%) | 0.3056 (3 seeds 0.305-0.310) | 0.591 (3-seed ensemble) |

Each doubling gives +0.007 then +0.004 per-game accuracy within this cache.
Extrapolation to substantially greater game diversity is unverified.
