# Hard Boundary Feature Atlas

Branch: `analysis/hard-boundary-feature-atlas`

This experiment is diagnostic, not a new rank model.

The goal is to answer:

> When v47 still confuses adjacent registered ranks, which existing feature families are genuinely indistinguishable, and which features still separate the players after conditioning on v47?

The atlas is intended to decide what to build next rather than trying another model blindly.

## Result (2026-11-18): do not build a larger classifier on these inputs

Run on the 2,000-question holdout3 cache: 52 features, 9 boundaries, 468 cells.
v47 replays 0.7640 (exact 0.6465, off by one 0.3195, off by two or more 0.034).

**Under a calibrated null and nonparametric conditioning, no officially
available feature family separates the hardest boundary.** The screen's own
instruction applies: no larger classifier on the same information. The one
family the atlas cannot yet see — a multi-profile HumanSL likelihood — is being
audited separately on the validation cohort.

### Where v47 is actually weak now

| Boundary | D/C | C/B | B/A | A/1D | **1D/2D** | 2D/3D | 3D/4D | 4D/5D | 5D/6D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v47 AUC | 0.908 | 0.913 | 0.919 | 0.834 | **0.777** | 0.836 | 0.865 | 0.921 | 0.906 |

4D/5D is listed as historically difficult, but that was the strength-score era
(`a1d_boundary` records 0.73). With opponent identity, v47 is at 0.921 there.
The genuinely hard boundaries are now 1D/2D, A/1D and 2D/3D.

### 1. The 0.58 cut is the noise ceiling of a family with no signal

The conditional AUC is direction-free, `max(AUC, 1 - AUC)`, with the direction
chosen on the evaluation data, then maximised over boundaries and over a
family. `permutation_null.py` shuffles labels within fold while holding the
label-free residuals fixed (2,000 draws), applying the same permutation to every
feature so the family-wise maximum keeps feature correlation:

| | Null q95 |
| --- | ---: |
| single cell, mean over 468 cells | 0.545 |
| best feature within a family, per boundary | 0.565 - 0.579 |

A family with no information reaches the 0.58 screen at its 95th percentile.
The v47 family itself stays under its null at every boundary under the atlas's
linear residual, which confirms the null is calibrated.

### 2. Linear residualization manufactures most of the signal

The atlas residualizes each feature on a straight line in the v47 boundary
log-odds. A fold-safe nonparametric alternative — subtract the feature's mean
within ten quantile bins of the same score, bins fitted on the opposite fold —
removes dependence a line cannot:

| | Linear | Binned |
| --- | ---: | ---: |
| cells with BH q < 0.10 | **50** | **13** |

Thirty-eight of the 50 linear survivors do not survive binning; twelve do,
and one cell (`base__entropy` @ D/C) appears only under binning.

| Feature @ boundary | Linear AUC (q) | Binned AUC (q) |
| --- | ---: | ---: |
| `opponent__soft_rank_0_mass` @ 5D/6D | 0.739 (0.008) | 0.618 (0.047) |
| `opponent__soft_rank_9_mass` @ D/C | 0.705 (0.008) | 0.626 (0.047) |
| `structure__hd1_opp_white_fraction` @ B/A | 0.642 (0.008) | 0.543 (0.646) |
| `structure__hd1_opp_white_fraction` @ A/1D | 0.614 (0.008) | 0.586 (0.117) |
| `base__top_probability` @ C/B | 0.633 (0.008) | 0.552 (0.468) |
| `structure__hd2plus_fraction` @ B/A | 0.744 (0.008) | 0.564 (0.308) |

The features that collapse completely are the ones with *low* Spearman
correlation to v47 (`hd1_opp_white_fraction` -0.19, `base__top_probability`
0.10). Low rank correlation does not mean independence: their relationship to
v47 is non-monotone, and a straight line leaves it in the residual.

### 3. What survives is v47's own information at easy boundaries

The 13 binned survivors are almost all `opponent__soft_rank_k_mass` at the
extreme boundaries (D/C, 5D/6D, 2D/3D, B/A). They shrink by about 0.1 but do
not vanish, and they should not be read as new information:

- their Spearman correlation with v47 expected rank is about 0.9 — they are
  v47's own opponent-rank posterior, in a direction orthogonal to one
  boundary's log-odds;
- binning is coarse (about 20 questions per bin), and the **v47 family itself
  beats the binned null at D/C** (0.578 against 0.568), so some v47 leakage
  remains at exactly the boundaries where these survive;
- most matched-pair effects straddle zero.

At most this suggests v47's decoding under-uses the tail of its own posterior
on easy boundaries. It is not an independent source.

### 4. The hardest boundary has nothing usable

At 1D/2D the binned survivors are `opponent__soft_rank_0_mass` (0.581, q 0.052,
matched effect 0.029 [-0.223, 0.316]) and `structure__hd2plus_fraction`, which
cannot help the official test set (no HD>=2 game there). A/1D and 2D/3D are the
same picture: only HD>=2 or a v47-redundant opponent mass clears the null.

### Recommendations for future atlas runs

1. Report the permutation null beside every conditional AUC; read 0.58 as
   noise, not as "interesting".
2. Prefer nonparametric residualization (`--residual binned`, or isotonic);
   38 of the 50 cells significant under the linear residual disappear.
3. Treat the matched-pair effect as confirmatory, since it conditions on v47
   without assuming a functional form.
4. Drop HD>=2 features from the candidate ranking automatically; they are a
   holdout3-only diagnostic.

### Fixes applied

- `build_features.py`: the per-game opponent-rank variance used a matrix
  product, `rank_prob @ (...)` on two `[batch, 10]` tensors. It failed for
  batch 1024 and would have silently mixed games for a batch of exactly 10. It
  is now the element-wise `sum_k p_k (k - mu)^2`.
- `analyze.py`: `Axes.boxplot(labels=...)` was renamed `tick_labels` in
  matplotlib 3.9; the venv has 3.11.2, where `labels` is rejected.

### Reproduce

    GPU=1 bash competition2_rank_prediction/experiments/hard_boundary_atlas/run.sh
    O=runtime/competition2_research/experiments/hard_boundary_atlas
    for mode in linear binned; do
      PYTHONPATH=.:competition1_player_identification \
        competition2_rank_prediction/c1/.venv/bin/python -m \
        competition2_rank_prediction.experiments.hard_boundary_atlas.permutation_null \
        --table $O/feature_table.npz --metrics $O/feature_metrics.csv \
        --residual $mode --output $O/permutation_null_$mode.json
    done

## Cohort

The first version uses the existing 2,000-question holdout3 opponent-identity cache:

`/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt`

It reconstructs cross-fitted v47 probabilities using the same two holdout3 halves. No official test labels are used.

## Built-in feature families

### Baseline / v47 confidence

- expected rank
- entropy
- top probability
- top-1 / top-2 margin
- expected-rank shift introduced by v47

### Opponent identity / matchmaking summaries

Per-game opponent identity and rank posterior are summarized to question level:

- top class cosine
- top identity posterior probability
- identity entropy
- expected opponent registered rank
- opponent-rank posterior standard deviation
- opponent-rank entropy
- top opponent-rank posterior
- mean / std / q25 / q75 across games
- soft opponent-rank mass for D through 6D

### Question structure

- number of games
- even-game fraction
- HD1 opponent-White fraction
- HD1 opponent-Black fraction
- HD>=2 fraction

HD>=2 is retained only as a holdout3 diagnostic. The official test contains no HD>=2 games, so it cannot be treated as an official opportunity.

## Optional feature sources

The builder accepts additional question-level NPZ files without code changes:

    EXTRA_NPZ=/path/humansl_profile.npz:/path/other_features.npz \
      bash competition2_rank_prediction/experiments/hard_boundary_atlas/run.sh

An extra NPZ needs one id field: `question_id`, `qid`, or `player_id`.

It can then provide either a 2-D `features` matrix plus `feature_names`, or ordinary numeric arrays whose first dimension is question count.

This is the intended hook for a paper-faithful multi-skill HumanSL likelihood profile later, including log-policy likelihoods, slope, curvature, low-vs-high policy contrasts and per-game variability.

## Analyses

### 1. Raw adjacent-boundary AUC

For every feature and each adjacent rank pair (D/C through 5D/6D), the atlas measures direction-free AUC. This answers whether a feature contains rank information at all.

### 2. v47-conditioned separability

Raw AUC is not enough: a feature can look excellent only because it restates the same strength signal v47 already knows.

For each boundary and target fold, the feature is residualized against the v47 high-vs-low log-odds using the opposite fold only. AUC is then measured on the untouched fold.

The resulting conditional AUC is the primary diagnostic.

### 3. Matched hard pairs

Within each fold, lower-rank and higher-rank questions are matched one-to-one by v47 boundary log-odds and question size. Only the closest half of pairs is retained by default.

For each feature the atlas reports the standardized matched difference `feature(higher rank) - feature(lower rank)` with a paired bootstrap 95% interval.

This asks the stricter question: if v47 thinks two players are nearly the same, what still changes systematically with registered rank?

### 4. Feature redundancy map

A scatter plot places every feature at:

- x = absolute Spearman correlation with v47 expected rank
- y = best conditional adjacent-boundary AUC

The desirable region is upper-left: high conditional discrimination with low redundancy.

### 5. Family atlas

For each feature family, the best conditional AUC is shown across all nine boundaries. This is the strategic plot for deciding whether the next investment should be opponent context, HumanSL multi-policy profile, question structure, or another independent source.

### 6. Detailed hard-boundary views

The atlas produces dedicated plots for the historically difficult boundaries:

- A / 1D
- 1D / 2D
- 4D / 5D

These show the strongest non-baseline features and their distributions inside hard/confused questions.

## Run

    git fetch origin
    git checkout analysis/hard-boundary-feature-atlas
    GPU=1 bash competition2_rank_prediction/experiments/hard_boundary_atlas/run.sh

The v47 refit is lightweight but the 24k x 6,807 similarity matrix is large, so a GPU is recommended for the feature-building pass. Plotting is CPU-only.

## Outputs

Primary outputs under `runtime/competition2_research/experiments/hard_boundary_atlas/`:

- `feature_table.npz`
- `feature_table.json`
- `feature_metrics.csv`
- `report.json`
- `report.html`
- `figures/00_v47_boundary_difficulty.png`
- `figures/01_raw_auc_heatmap.png`
- `figures/02_conditional_auc_heatmap.png`
- `figures/03_hard_pair_effect_heatmap.png`
- `figures/04_family_conditional_auc_heatmap.png`
- `figures/05_redundancy_vs_conditional_auc.png`
- detailed A/1D, 1D/2D and 4D/5D plots

## Interpretation gate

This is a research-priority screen, not a production acceptance threshold.

A feature family becomes interesting if, especially on A/1D, 1D/2D or 4D/5D, conditional AUC is roughly 0.58 or higher, preferably 0.60+, with a consistent matched-hard-pair effect and low/moderate correlation with v47.

If every non-baseline family remains around 0.50-0.54 after conditioning, the next step should not be a larger classifier on the same inputs.

If one family remains clearly separable, the next model should be a small conditional specialist built specifically around that information source.
