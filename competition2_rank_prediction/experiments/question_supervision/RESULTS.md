# End-to-end question supervision — completed 2026-11-09

## Decision: do not replace original C1+C2

The matched own-branch pilot passed, but its benefit did **not** survive
the complete C1+C2 comparison. This is not a breakthrough or a new Public
best. Preserve the original reported Public0.7227214 submission.

## Preregistered matched pilot

Original decision Transformer, no added parameters. Train1,400 players
(28,000games), inner300, evaluation300; every partition player-disjoint.
Normalization is recomputed from dense optimizer games only. Seeds0/1/2,
40epochs ×64updates each, same sampled bags/augmentation/initialization.

- A: equally bag-weighted per-game NLL.
- B:0.5 × A +0.5 × differentiable whole-question NLL, using production
  log-mean pooling and its probability floor.
- Select earliest maximum inner official query utility, never evaluation.

| Mean over three fitted seeds | A | B |
| --- | ---: | ---: |
| Official utility, own branch only | 0.607895 | **0.631648** |
| Question NLL | 1.527448 | **1.484694** |

Delta+0.023754, paired-player95% interval[+0.004789,+0.042706]. Seed changes:
+0.029844,+0.026321,+0.015095. Worst rank delta−0.011111, so all
preregistered screening conditions pass. Bootstrap first averages seed
changes within each player;300 players, not900 independent observations.
Only one fixed question per evaluation player is used in this pilot.

Game-count5–9/10–14/15–20 deltas:+0.029110/+0.012090/+0.028302.
Largest rank changes:2D+0.087219,6D+0.096545. These subgroup observations
are descriptive, not independent significant discoveries.

## Conditional full-pipeline check

Original400 validation players are absent from neural optimizer AND inner
selection partitions. Predict their8,000 games with each three-seed A/B
ensemble; replace d96 only, freeze original ResNet and opponent C2.
Refit geometric fusion and C1 in five rank-stratified player-disjoint
calibration folds, exactly as in earlier production comparisons. C1's
own/opponent features use the corresponding replaced own model.

| Full C1+C2 pipeline | Calibration-CV utility |
| --- | ---: |
| Original frozen production models | **0.710606** |
| Replace d96 with matched dense A | 0.711483 |
| Replace d96 with question-supervised B | 0.710430 |

B−original:−0.000176,97.5% player interval[−0.004811,+0.004459].
B−A:−0.001053,97.5% player interval[−0.004111,+0.001909].
The original reference replays exactly0.7106057000201695. Both contrasts
are inconclusive, and no full-pipeline benefit is established.

The400-player calibration cohort and300-player pilot cohort were studied
historically. Both checks are exploratory, not fresh confirmation. None
of these numbers is an official Public score or the2,000-player0.717012.

## What this changes about the search

1. Whole-question supervision is trainable and useful for its own branch;
   frozen embeddings were not the only possible training scheme.
2. Better own-branch utility is insufficient as a production selection
   criterion. The gain may overlap with information already recovered by
   opponent C2/C1, or alter calibration in a way that cancels its benefit.
   This experiment does not prove which explanation is causal.
3. Do not broaden pooling weights/seeds based on the outer score. The next
   direction should first establish *conditional* information beyond all
   existing members. Richer upstream board representations are a candidate
   to investigate, not a demonstrated solution.
4. Goal "find a breakthrough" remains unfulfilled. This experiment is
   completed, documented and rejected as a production replacement.

## Reproduction and verification

See [README.md](README.md) and [fixed plan](PLAN.md). Artifacts are in
`production/question_supervision_20261109/`: source-hash manifest, audited
splits, all6 checkpoints/histories, per-seed evaluation predictions,
`report.json`, `learning_curves.png`, `confirmation_inputs.npz`,
`confirmation_predictions.npz`, `confirmation_report.json`.

- pylint on experiment directory:10.00/10, no disables.
- mypy `--follow-imports=silent --ignore-missing-imports`:7files pass;
  imported legacy modules/missing third-party stubs are outside this scope.
-5unit tests pass: inference-equivalent pooling, clipping floor,
  encoder-reaching gradients, balanced bag weighting and invalid inputs.
- An identical-control evaluation returns zero delta and fails advancement.
- Original submission SHA-256 unchanged:
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
- Training/evaluation terminal; no new official submission generated.
  GPU0 vLLM and the other session's soft-boundary jobs were not modified.
