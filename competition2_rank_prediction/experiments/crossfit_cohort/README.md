# 2,000-player cross-fitted confirmation (2026-10-21)

Protocol fixed in advance in `PREREGISTRATION.md`; scored once by `evaluate.py` (`results/report.json`, `logs/evaluate.log`).
Cohort `features/holdout3` (2,000 new players, 200 per rank, 24,822 games), two rank-balanced halves; models `*_c{j}`
trained without half j score only half j; everything else fitted on the validation cohort. The baseline reproduced
`predict_test.py`'s decisions exactly on every question (asserted).

| method | score (2,000 q) | diff vs default [95%] | one-sided p | Holm |
| --- | --- | --- | --- | --- |
| default: best2 + borrowed opponent member | 0.6763 | - | - | - |
| C1 shift residual (E1+E2+E3, lam 0.01) | 0.7061 | +0.030 [+0.017, +0.043] | < 0.0001 | adopt (thr 0.025) |
| C2 opponent-specific Transformer (from scratch) | 0.6833 | +0.007 [+0.001, +0.013] | 0.008 | adopt (thr 0.050) |
| C1 + C2 (exploratory) | 0.7170 | +0.041 [+0.027, +0.054] | < 0.0001 | - |

Halves: default 0.6933 / 0.6594, C1 0.7145 / 0.6977, C2 0.7022 / 0.6645, C1+C2 0.7265 / 0.7075 (both halves agree).
Per-rank change of C1: large gains at D (+0.10), 3D (+0.09), 4D (+0.10), 1D (+0.05), A (+0.04); losses at C (-0.09) and
5D (-0.05). C2 is small but nearly uniform (A +0.016, 1D +0.021).
The shift residual was not supported by validation CV earlier (opponent_meta) yet is confirmed here with a 5x larger sample:
the validation CV estimate was too noisy, which is exactly why the large cohort was needed.
