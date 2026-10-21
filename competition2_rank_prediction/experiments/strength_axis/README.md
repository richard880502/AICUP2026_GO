# Strength axis vs richer representations (2026-10-20)

Question: does the model lack rank information, or is the cut into 10 ranks the problem?
All calibrators are fitted on player-disjoint out-of-fold validation questions (400 players, 20 simulated 5-20 game questions each,
5 folds by player). The untouched holdout is scored once with calibrators fitted on all validation players; nothing is tuned on it.
Model: holdout-excluded dense Transformer `d96_h` (3 seeds). s = the ordinal head's strength score, standardised per seed, averaged
over seeds and over the question's games. D uses seed 0's attention-pooled 128-d game vector averaged over the question.

| method | CV score | CV exact | CV vs reference (95%, player bootstrap) | holdout score | holdout exact |
| --- | --- | --- | --- | --- | --- |
| A production best2 (dense Transformer + cosine ResNet) | 0.6349 | 0.480 | - | 0.6634 | 0.530 |
| A2 same pipeline, dense Transformer only | 0.6215 | 0.463 | vs A: -0.0132 [-0.0253, -0.0010] | 0.6498 | 0.510 |
| B s + 9 hard cut points (maximise score) | 0.6109 | 0.449 | vs A2: -0.0106 [-0.0269, +0.0062] | 0.6397 | 0.492 |
| C s (+ log k) + ordinal calibration | 0.6182 | 0.461 | vs A2: -0.0034 [-0.0140, +0.0076] | 0.6378 | 0.492 |
| D 128-d mean (+ s, log k) + L2 ordinal (l2 = 1e-3 by CV) | 0.6299 | 0.477 | vs A2: +0.0084 [-0.0110, +0.0279] | 0.6435 | 0.500 |

Conclusions:
1. One scalar strength score carries almost all of the Transformer's rank information: C is within noise of A2 in CV (-0.003) and
   1.2 points lower on the holdout.
2. How the axis is cut is not the bottleneck: hard cut points (B) and a calibrated ordinal model (C) score about the same.
3. The 128-d player vector does not add reliable information beyond the Transformer's own probabilities (D vs A2: CV +0.008 with an
   interval crossing zero, holdout -0.006).
4. The only reliable gain comes from another information source: adding the ResNet (A vs A2) is +0.013 in CV (interval excludes 0)
   and +0.014 on the holdout.
=> What is missing is discriminative information (neighbouring ranks overlap on the strength axis), not a better cut. Gains should come
   from independent evidence sources, not from re-reading the same Transformer. Caveat: D averages per-game vectors; a model trained
   jointly across games (the player-bag candidates in CURRENT_BEST.md) is a different test.
Reproduce: `c1/.venv/bin/python experiments/strength_axis/run.py` (writes results/report.json).

## Experiment A: ranking ability with d96_h, three seeds, real 5-20 game questions (2026-10-20)
`ranking_ability.py` (uses `results/questions.npz` written by `run.py`, i.e. exactly the questions of the method comparison).
Figures: `figures/strength_by_rank.png` (score distribution per true rank, by question size), `figures/adjacent_overlap.png`
(adjacent-rank AUC = P(a random question of the higher rank scores higher), 0.5 = complete overlap). Numbers: `results/ranking_ability.json`.

| | validation questions (8,000) | holdout questions (400) |
| --- | --- | --- |
| Spearman(strength, rank), all | 0.938 | 0.931 |
| by games 5-9 / 10-14 / 15-20 | 0.931 / 0.939 / 0.943 | 0.947 / 0.942 / 0.905 |
| adjacent AUC (val) | D/C 0.80, C/B 0.81, B/A 0.84, **A/1D 0.61**, 1D/2D 0.72, 2D/3D 0.78, 3D/4D 0.82, **4D/5D 0.73**, 5D/6D 0.82 | A/1D 0.84, **1D/2D 0.57**, 2D/3D 0.71 (40 per rank, noisy) |

Score / exact by question size, same questions (experiment B read off the same rows):

| games | A best2 | A2 dense only | C axis + calibration | D 128-d |
| --- | --- | --- | --- | --- |
| val 5-9 | 0.614 / 0.454 | 0.604 / 0.443 | 0.597 / 0.433 | 0.605 / 0.446 |
| val 10-14 | 0.641 / 0.486 | 0.628 / 0.471 | 0.627 / 0.474 | 0.637 / 0.488 |
| val 15-20 | 0.648 / 0.497 | 0.631 / 0.473 | 0.629 / 0.473 | 0.645 / 0.496 |

Findings:
1. The 0.94 ranking ability survives with the holdout-safe model and real question sizes (0.938 val, 0.931 holdout).
2. More games barely help: Spearman 0.931 -> 0.943 from 5-9 to 15-20 games, and the within-rank spread of the score hardly narrows
   (box widths are similar across the three panels). The spread is between players of the same rank, not per-game noise.
3. Overlap is concentrated at specific boundaries: A/1D is the worst (AUC about 0.6, i.e. A and 1D players are nearly
   indistinguishable on this axis and the median of 1D is barely above A), then 1D/2D and 4D/5D. Most other pairs are 0.78-0.86.
4. Decision table: the single axis (C) is close to the same Transformer's full pipeline (A2) at every question size, trails best2 by
   about 0.015-0.02 (= the ResNet's contribution), and both show heavy adjacent overlap -> priority is better input evidence,
   game sampling and label quality, not calibration. D (128-d) tracks best2 on validation (+0.01-0.016 over C at every size), so the
   128-d vector may carry a little information beyond the scalar, but this was not confirmed on the holdout.
