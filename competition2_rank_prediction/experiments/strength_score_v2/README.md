# Strength Score v2 — HumanSL relative-strength experiment

## Goal

Test whether a **continuous relative playing-strength signal** learned directly
from frozen HumanSL rich trunk activations adds information beyond the current
Competition 2 pipeline.

This is different from `experiments/strength_axis`: that experiment analyses
the scalar axis already produced by the d96 rank Transformer. Here the scalar
is learned independently from Competition 1's 960-D HumanSL `style_v7`
representation.

## Model

For every valid own move:

```text
960-D HumanSL rich trunk
  -> LayerNorm
  -> Linear 960 -> 256 -> GELU
  -> Linear 256 -> 64 -> GELU
  -> Linear 64 -> 1
  -> beta_move
```

Then:

```text
mean(beta_move) -> beta_game
mean(beta_game over question/player bag) -> beta_player
```

Training is balanced by player and rank. The primary objective is a
Bradley-Terry-style relative-order loss: higher registered ranks should usually
receive higher beta. Rank-distance weighting makes distant-rank comparisons
stronger supervision than noisy adjacent boundaries such as A/1D. A small
within-player game-consistency loss is added.

## Validation protocol

Two holdout3 folds are trained independently with the corresponding
`exclude_c0.txt` / `exclude_c1.txt` players removed.

The final evaluator can rebuild the current v4+v7 opponent-identity cross-fit
baseline from `h3_sims_v47.pt`. For each target fold, the Strength Score
fusion weight is selected on the **opposite** fold.

Default go/no-go gate:

```text
fused score - current opponent-identity baseline >= +0.008
```

Also inspect:

- Strength Score Spearman correlation with true rank
- adjacent-rank AUC, especially A/1D
- correlation with the baseline expected rank
- rank-wise mean beta ordering

If the gain is only +0.001 to +0.003, treat it as insufficient evidence and do
not expand this branch into a larger Set Transformer.

## Existing artifacts reused

Training does **not** rerun HumanSL over the million-game rank corpus. It
streams the already-generated Competition 1 rich-token shards:

```text
production/trunk_style_20261113/corpus_rich/
production/trunk_style_20261113/corpus2_rich/
production/trunk_style_20261113/rich_pca.npz
```

`prepare_cache.py` builds a compact mmap cache with defaults of 12
games/player and 48 sampled moves/game.

Holdout3 own-side games are extracted once with the existing
`extract_rich.py` pipeline.

## Run

From repository root:

```bash
git checkout experiment/strength-score-v2

DEVICE=cuda:0 GPU=0   bash competition2_rank_prediction/experiments/strength_score_v2/run.sh
```

The runner expects:

```text
runtime/competition2_research/features/holdout3/
production/trunk_style_20261113/corpus_rich/
production/trunk_style_20261113/corpus2_rich/
production/trunk_style_20261113/rich_pca.npz
```

For fusion it defaults to:

```text
/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt
```

Override when needed:

```bash
OPP_STATE=/path/to/h3_sims_v47.pt DEVICE=cuda:0 GPU=0   bash competition2_rank_prediction/experiments/strength_score_v2/run.sh
```

## Outputs

```text
competition2_rank_prediction/experiments/strength_score_v2/results/
  fold0/
    checkpoint.pt
    predictions.npz
    report.json
  fold1/
    checkpoint.pt
    predictions.npz
    report.json
  crossfit_report.json
```

The final report contains approximately:

```json
{
  "strength_only": {
    "score": 0.0,
    "spearman": 0.0,
    "adjacent_auc": {"A/1D": 0.0}
  },
  "current_baseline": {"score": 0.0},
  "independence": {
    "spearman_strength_vs_baseline_expected_rank": 0.0
  },
  "fusion": {
    "score": 0.0,
    "delta_vs_current_baseline": 0.0,
    "passes_gate": false
  }
}
```

## Fast plumbing test

For a small cache:

```bash
PYTHONPATH=.:competition1_player_identification competition2_rank_prediction/c1/.venv/bin/python -m   competition2_rank_prediction.experiments.strength_score_v2.prepare_cache   --max-players-per-rank 100   --games-per-player 8   --moves-per-game 32
```

Use reduced epochs/steps for a smoke run. Do not use that pilot score for the
final gate.
