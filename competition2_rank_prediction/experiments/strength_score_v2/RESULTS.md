# Strength Score v2 — no gain; the scalar is redundant with C1+C2

Completed 2026-11-17. Exploratory holdout3 cross-fit, not a Public result and
not a new submission. **The fusion weight selected on the opposite fold is
zero on both folds, so the fused prediction is identical to the current
opponent-identity baseline.** The advancement gate (+0.008) fails.

## Headline

| Condition | Strength alone | Fused | Baseline | Delta | Weights (fold0/fold1) |
| --- | ---: | ---: | ---: | ---: | --- |
| `forward` (as designed) | 0.634820 | 0.764354 | 0.764354 | **0.000000** | 0.0 / 0.0 |
| `uniform` | 0.633556 | 0.760405 | 0.763721 | -0.003316 | 0.0 / 0.1 |
| `inverse` | 0.634315 | 0.764354 | 0.764354 | 0.000000 | 0.0 / 0.0 |
| `adjacent` | 0.641452 | 0.764670 | 0.764670 | 0.000000 | 0.0 / 0.0 |

The baseline differs by up to 0.0010 between rows because the opponent model
is refitted per run; three independent replays of it in this session gave
0.7644 / 0.7647 / 0.7639, so that spread is refit noise, not a condition
effect.

## The model works; it is simply not new information

The scalar is well behaved. Rank-wise mean strength is perfectly monotone
over all ten registered ranks, and Spearman correlation with the true rank is
0.943:

```text
D -3.06  C -2.32  B -1.61  A -0.78  1D -0.22
2D 0.13  3D 0.60  4D 1.07  5D 1.68  6D 2.25
```

A single scalar scores 0.6348 on its own. The problem is independence:

| | Pearson | Spearman |
| --- | ---: | ---: |
| Strength vs baseline expected rank | **0.957** | 0.966 |

A different representation (960-D HumanSL rich trunk instead of the 41 engine
scalars), a different objective (Bradley-Terry ordering instead of ordinal
classification) and roughly 18x the per-player data still measure the same
quantity the production pipeline already measures: **playing strength**.

## Why the default weighting was questioned, and the ablation that settles it

After the `forward` run returned exactly zero, the remaining holdout3 error
was decomposed. Against the v47 baseline, 64.7% of questions are exact, about
33% are off by exactly one rank, and only about 2.3% are off by two or more.
Distant-rank discrimination is already solved; adjacent-rank discrimination is
the only thing left to win.

The default `distance_weight = |gap|.clamp(max=3) / 3` gives an adjacent pair
one third of the weight of a distant pair, which is the opposite of where the
remaining error sits. `--distance-weighting` was added to test this
(`forward` reproduces the original behaviour and stays the default).

Adjacent-boundary AUC, the quantity the ablation targets:

| Weighting | A/1D | 1D/2D | D/C | Spearman |
| --- | ---: | ---: | ---: | ---: |
| `forward` | 0.7454 | **0.6725** | 0.7890 | 0.9430 |
| `uniform` | 0.7512 | **0.6740** | 0.7934 | 0.9451 |
| `inverse` | 0.7626 | **0.6794** | 0.8000 | 0.9501 |
| `adjacent` (one-rank pairs only) | 0.7571 | **0.6796** | 0.8089 | 0.9468 |

Moving every unit of supervision onto one-rank pairs raises 1D/2D AUC from
0.6725 to 0.6796, and A/1D from 0.7454 to 0.7571. The fusion weight stays
zero. The independence correlation stays in 0.955-0.965 in all four
conditions.

**Conclusion: the objective was not the limitation.** Adjacent registered
ranks are not separable from the player's own moves, however the loss is
weighted. This is consistent with the recorded win-rate evidence in
[the opponent identity record](../opponent_identity/RESULTS.md): among
confidently identified opponents in even games the asked player wins 45% when
registered one rank below and 59% when one rank above, so a one-rank
registered difference carries far less than one rank of playing strength.

## What was run

- `prepare_cache` over the Competition 1 rich-token shards: 7,983 players,
  93,446 games, 48 moves per game, rank-balanced sampling at training time.
  No HumanSL re-run over the rank corpus.
- `prepare_holdout3` + `extract_rich`: 2,000 questions / 24,822 own-side
  games, 160 samples per game, 24,822 valid and 0 empty.
- Two folds per condition, players in `exclude_c{fold}.txt` removed, 24
  epochs x 160 steps. Each fold trains in about 90 seconds on one 3090.
- `evaluate_crossfit` rebuilds the v4+v7 opponent baseline from
  `h3_sims_v47.pt` and selects the fusion weight on the opposite fold.

## Fix applied

`prepare_eval_cache.py` contained a literal newline where `"\n"` was intended,
so the module could not be imported at all. Verified against a fresh clone
that this was committed, not a transfer artifact. The fix only affects the
trailing newline of `report.json`.

## Decision

Keep the current production pipeline. Per this README's own rule, a gain of
+0.001 to +0.003 would be insufficient evidence to expand the branch into a
larger Set Transformer; the measured gain is 0.000, so the branch is not
expanded. No official-test inference, no new submission, no upload.
