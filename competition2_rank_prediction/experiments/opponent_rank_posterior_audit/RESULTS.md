# Opponent rank posterior audit (P0) — results

**Bottom line (2026-11-20): v47's sum-over-classes pooling is the right one.** The class-count correction
that looked promising on the control (+0.0156) costs v47 0.0146 on holdout3 (see the P1 section at the
end). Keep production unchanged.

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

## P1 inside v47: the count correction hurts — do not adopt

`count_prior_v47.py`, preregistered before running (arms, beta, gate and jitter all fixed in its docstring). Same
two-half holdout3 cross-fit and deterministic algorithms as the ten-profile P2. The correction scales the per-game
opponent-rank posterior by `n_r^-beta` before v47's unknown mixture; `n_r` counts the classes left in the softmax
after masking holdout3 players, which makes the imbalance even sharper (A 1,265, 1D 194).

Sanity: `beta0` runs the same code path with the correction skipped and replays production exactly (0.765302, max
abs difference 0.0).

| Arm vs `beta0` | Delta | 95% CI | Halves | Changed (better / worse) |
| --- | ---: | --- | --- | --- |
| **`beta1` (primary, fixed from the control)** | **-0.0146** | [-0.0218, -0.0075] | -0.0168 / -0.0124 | 152 (50 / 100) |
| `beta_learned` (fit 0.154 / 0.301 by half) | -0.0086 | [-0.0130, -0.0044] | -0.0060 / -0.0112 | 55 (13 / 41) |
| `beta1`, 1,587 official-compatible questions | -0.0154 | [-0.0234, -0.0071] | -0.0167 / -0.0142 | 119 (38 / 80) |
| `beta_learned`, official-compatible | -0.0074 | [-0.0124, -0.0028] | -0.0039 / -0.0108 | 44 (12 / 31) |

Five jittered refits (base times `exp(1e-6 N(0,1))`, shared across arms): `beta1 - beta0` = -0.0138, -0.0140,
-0.0121, -0.0139, -0.0130 — mean **-0.0134**, SD 0.0008, far outside refit noise. Gate: **fails** on every criterion.

The correction was meant to help 1D. It hurts 1D most (per-rank delta **-0.0348**), and **1D/2D AUC falls from 0.7773
to 0.7548**. Learned beta moves away from zero because training-half NLL prefers a little correction, yet utility
still drops: NLL and expected-utility decoding disagree here.

### Why the control misled

`sum_{c in r} P(c | game)` is the Bayes marginal over opponent ranks under a uniform prior over **players**, which is
the right prior when opponents are drawn from the player population: more A players means more A opponents. The
control was sampled to be close to rank-balanced, so a uniform prior over **ranks** (beta = 1) matched that sample's
design rather than real opponents. A rank-balanced control is fine for measuring retrieval (the 26% same-rank share,
soft-pooling calibration) and wrong for choosing a prior.

`count_prior_v47.py`: pylint 10.00, mypy clean. Run from `runtime/competition2_research`:

    CUDA_VISIBLE_DEVICES=1 c1/.venv/bin/python -m \
      experiments.opponent_rank_posterior_audit.count_prior_v47 \
      --output /mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit/count_prior_v47.json

## P2 diagnostic: v47 error-source decomposition

Run 2026-11-20 with `run_decomposition.sh` (`decompose_v47.py`); the seven bundled unit tests pass. Exploratory and
not causal: there are no true opponent ranks on holdout3, so no oracle attribution is possible.

Sanity: the full arm replays v47 exactly (0.7653, absolute difference 0.0) and the base replays C1+C2 (0.7170).

### Nearly all of v47's gain over C1+C2 is identity retrieval

| Arm | Utility | Exact | NLL | 1D/2D AUC | A/1D AUC | 2D/3D AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C1+C2 base | 0.7170 | 0.5820 | 1.0790 | 0.7257 | 0.8136 | 0.8050 |
| v47 refitted without game-specific identity | **0.7111** | 0.5755 | 1.0457 | 0.7194 | 0.8175 | 0.7914 |
| full v47 | **0.7653** | 0.6485 | 0.8788 | **0.7773** | 0.8337 | 0.8341 |

Retrieval increment (full minus refitted no-retrieval): **+0.0542 [+0.0403, +0.0690]**, halves +0.0568 / +0.0517;
+0.0486 [+0.0330, +0.0643] on the 1,587 official-compatible questions. Without identity, the rest of the v47 stack
(game kind, game count, transition, C1+C2 base) scores **below** C1+C2 alone, so its value is entirely the retrieved
opponent ranks. That holds at the hardest boundary too: 1D/2D AUC 0.719 without retrieval, 0.777 with it.

Adding the whole identity stack to the base: 972 questions right in both, 325 base errors rescued, 192 base-correct
questions hurt (mostly to an adjacent rank; their full-arm utility is 0.364), 511 wrong in both.

### v47's two knobs sit at their optimum

| Fixed same-model perturbation | Delta vs full | 95% CI |
| --- | ---: | --- |
| retrieval temperature x1.25 (softer) | -0.0093 | [-0.0165, -0.0018] |
| retrieval temperature x0.75 (sharper) | -0.0092 | [-0.0153, -0.0030] |
| transition logits x0.75 (softer) | -0.0120 | [-0.0194, -0.0037] |
| transition logits x1.25 (sharper) | -0.0160 | [-0.0238, -0.0083] |

Every direction hurts, consistent with the temperature sweeps in the P0 and count-correction work.

### Most remaining errors do not respond to either knob

Official-compatible questions v47 gets wrong (567), grouped by which ±25% perturbation flips the chosen rank:

| Flipped by | Wrong questions | High / low confidence |
| --- | ---: | --- |
| retrieval only | 29 | 8 / 21 |
| transition only | 51 | 21 / 30 |
| both | 66 | 23 / 43 |
| **neither** | **421 (74%)** | 187 / 234 |

Sensitivity, not attribution, and these perturbations are not matched in strength. Still, three quarters of the
remaining errors are not near a decision boundary of either component. That fits what the rest of this project
found: the residual error comes from registered ranks that disagree with playing strength and from the width of
matchmaking itself, not from mis-tuned retrieval or transition.

Confidence bins (top-rank mass of the opponent rank posterior, opposite-fold median 0.517): utility 0.722 on the
low-mass half, 0.806 on the high-mass half.

### Instability worth knowing

The fitted v47 parameters differ a lot between the two halves: the HD1 opponent-White unknown share is 0.421 in one
and 0.089 in the other, and the base weight is 0.48 against 0.71. This matches the refit sensitivity measured in the
ten-profile P2 and is one reason single-refit differences under about 0.004 are not reliable on this cohort.

### Reading

No cheap v47 lever is left. Retrieval is what makes v47 work, its temperature and the transition strength are
already at their optimum, the class-count prior is already right (P1 above), and most remaining errors sit away from
the boundaries those components control.
