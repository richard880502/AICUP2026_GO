# Hard Boundary Feature Atlas

Branch: `analysis/hard-boundary-feature-atlas`

This experiment is diagnostic, not a new rank model.

The goal is to answer:

> When v47 still confuses adjacent registered ranks, which existing feature families are genuinely indistinguishable, and which features still separate the players after conditioning on v47?

The atlas is intended to decide what to build next rather than trying another model blindly.

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
