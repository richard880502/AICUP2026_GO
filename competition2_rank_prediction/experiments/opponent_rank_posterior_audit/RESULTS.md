# Opponent rank posterior audit (P0) — results

Completed 2026-11-20. Known-player control only: 4,096 unused games from 1,754 known v4 gallery players, 6,807 classes.
Not a v47 or competition utility, and not representative of the ~25% of official opponents who are not training
players. No training; no official test record read. The five bundled unit tests pass.

Cross-check: identity top-1 is **0.4182**, identical to `p_correct` recorded in `controls.npz` for this sample.

## Main results

| Rank retrieval | Top-1 | Within one | Rank NLL | Brier |
| --- | ---: | ---: | ---: | ---: |
| identity top-1 (identity itself) | 0.4182 | - | - | - |
| hard: rank of the top-1 class | 0.5701 | 0.7761 | 11.8795 | 0.8599 |
| soft, T = 0.04 (v47 initial) | 0.5620 | 0.7815 | 1.1851 | 0.5553 |
| soft, T cross-fitted (0.03 on both folds) | 0.5735 | 0.7874 | 1.1713 | 0.5486 |

- **26.1% of identity errors still carry the correct rank** (622 of 2,383), which is why rank retrieval (0.570)
  exceeds identity retrieval (0.418).
- Soft pooling is far better calibrated than a hard pick (NLL 1.19 against 11.88) but its argmax is not better:
  soft minus hard top-1 = -0.0081 [-0.0189, +0.0032].
- A sharper temperature helps slightly: tuned minus fixed top-1 +0.0115 [+0.0039, +0.0192], NLL improvement +0.0139
  [+0.0010, +0.0259]. v47 already learns its temperature (about 0.036 in P2's refits), so most of this is likely
  captured.
- Soft rank AUC per single game is fairly high at the hard boundaries (A/1D 0.892, 1D/2D 0.882, 2D/3D 0.851); the
  weakest are 4D/5D (0.832) and 5D/6D (0.833).
- Identity is harder for stronger players (top-1 0.59 at D down to 0.26 at 5D), yet hard rank top-1 stays 0.45-0.51
  across 1D-5D.
- Low vs high cosine-margin games (cut from the opposite fold): rank accuracy 0.40 vs 0.73. Soft does not beat hard
  in either bin (-0.008 in both).

By the README's pre-declared reading: the same-rank share of identity mistakes is low-ish (26%) and calibrated rank
NLL stays high (1.17 against 2.30 for uniform), so a controlled rank-aware follow-up is justified.

## Exploratory follow-up: soft pooling over-weights populous ranks

The confusion matrix shows soft pooling pulling true 1D, 2D and 3D games toward A (true 1D predicted A: 100 hard,
176 soft at T = 0.04). `softmax(cos / T) @ one_hot(rank)` **sums** probability over every gallery class of a rank, and
the v4 gallery is unbalanced:

| | D | C | B | A | 1D | 2D | 3D | 4D | 5D | 6D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gallery classes | 746 | 697 | 906 | **1,434** | **358** | 481 | 532 | 531 | 532 | 590 |
| control games | 398 | 441 | 384 | 317 | 481 | 394 | 471 | 418 | 405 | 387 |

`count_prior_check.py` scores `posterior_r ∝ n_r^(-beta) · Σ_{c∈r} exp(cos_c / T)` with T and beta chosen on the
opposite player-disjoint fold by rank NLL from fixed grids. Both folds choose **T = 0.03, beta = 1.0** (average rather
than sum within a rank).

| | Top-1 | Within one | NLL | true 1D -> A |
| --- | ---: | ---: | ---: | ---: |
| uncorrected, T tuned | 0.5735 | 0.7874 | 1.1713 | 137 |
| count-corrected | **0.5891** | 0.7957 | **1.1190** | **46** |

Corrected minus uncorrected top-1: **+0.0156, 95% CI [+0.0066, +0.0245]** (1,754 player clusters, game-weighted).

| Per-rank top-1 | D | C | B | A | 1D | 2D | 3D | 4D | 5D | 6D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| uncorrected | 0.807 | 0.660 | 0.651 | 0.685 | 0.451 | 0.444 | 0.467 | 0.498 | 0.457 | 0.685 |
| corrected | 0.824 | 0.692 | 0.630 | **0.524** | **0.599** | 0.510 | 0.484 | 0.505 | 0.462 | 0.664 |

### Do not over-read this

The correction is equivalent to a uniform prior over ranks, and **this control sample is close to rank-balanced**
while the gallery is A-heavy. Part or all of the gain may be prior-matching to this sample rather than better rank
information: A loses 0.161 while 1D gains 0.148.

It may still matter for v47. holdout3 and the official questions ask about rank-balanced players, and matchmaking
pairs opponents near the asked player's rank, so real opponents are plausibly closer to balanced than to the
gallery. But v47's learned transition tables and bias may already absorb a constant per-rank skew. **The only valid
test is inside v47**: add the count correction (or nine per-opponent-rank log offsets) to the posterior, refit under
the same cross-fit, and read the paired question-level delta against refit noise (about ±0.001 measured in the
ten-profile P2). Per this README, that belongs in a properly gated P1, not a decision from this P0.

## Fixes applied

- `run.sh`: `PYTHONPATH` held only the repository root, but `competition1_player_identification` modules import their
  siblings by top-level name (`cache_rich_profiles` imports `evaluate_decision_features`), so `score_present.py`
  failed with `ModuleNotFoundError`. That directory is now on the path too.
- `score_present.py`: the result's `game_id` failed the audit's duplicate check. Training `game_id`s are unique only
  within one rank file: 76 ids in `present.csv` are shared by two different games from different rank files, with
  different SGFs and different players. Each id is now qualified with its global training row from `note`, so the
  uniqueness check still guards real duplicates. Alignment with the token shards is positional and unaffected.
- The token cache was re-extracted (the original `present/` folder had been deleted to free disk): 4 shards, 4,096
  games, about 9.8 games/s.

`count_prior_check.py`: pylint 10.00, mypy clean.
