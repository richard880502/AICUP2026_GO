# Coverage pilot: exploratory screen passes

Completed 2026-11-09. These are **own-branch internal validation** scores,
not complete C1+C2 and not Public. The 400 validation players have been
historically explored; this is not independent confirmation.

| Fixed-budget three-seed probability ensemble | Official utility |
| --- | ---: |
| Two games per optimizer player | 0.6064277776 |
| Eight games per optimizer player | 0.6192710338 |

Delta +0.01284326; paired-player 95% bootstrap interval
[+0.00021178, +0.02593740]. Seed-specific deltas:
-0.00562688, +0.02752453, +0.01847597. The lower bound is only narrowly
positive and one seed regresses. This passes the fixed pilot rule but is not
a robustly established production gain.

Query-size deltas: 5–9 games +0.01295234; 10–14 +0.01861158;
15–20 +0.00780006. Worst rank delta is 2D -0.04331588, within the fixed
-0.05 safety threshold but important to monitor. Do not hide this tradeoff.

## Execution and checks

The complete 4,000-game cap8 extraction finished normally on GPU1, eight
500-game shards, with throughput approximately 5.47–6.34 games/s. The earlier
16-game timing was a poor sustained-throughput estimate. Both arms reused
identical feature values for shared games. Four hundred optimizer players,
100 inner players, and the original 400 evaluation players are disjoint.
All six runs completed 40 x 64 updates; only within-player coverage changed.

Selected epochs for cap2 seeds0/1/2: 2/4/3; cap8: 6/9/6. Shared cap2-only
normalization was identical across all checkpoints. Independent checks verified
all recorded source/input hashes, minimum-inner-NLL checkpoint selection,
optimizer/inner/evaluation player isolation, and unchanged original submission
SHA256. Pylint10/10 and mypy passed for changed modules; two data-guard tests
and CPU gradient / synthetic scoring checks passed.

Artifacts: `production/game_coverage_pilot_20261109/`, including eight feature
shards and `training/` with six checkpoints, histories, validation predictions,
`report.json`, `input_hashes.json` and `learning_curves.png`.

## Decision

The planned full C1+C2 comparison is now complete and **does not support
production replacement**:

| Full five-fold player-disjoint calibration CV | Utility |
| --- | ---: |
| Original C1+C2 | 0.7106057000 |
| Replace d96 with cap2 pilot ensemble | 0.7095007754 |
| Replace d96 with cap8 pilot ensemble | 0.7076199411 |

Cap8 minus original: -0.00298576, paired-player97.5% interval
[-0.00968780,+0.00330277]. Cap8 minus cap2: -0.00188083,
interval [-0.00851852,+0.00458482]. Two planned comparisons use97.5%
intervals. Other branches remain frozen; fusion and C1 fit only on the other
four player folds. Original replay matches exactly. Saved prediction-score
and bootstrap replay matches the report exactly. The confirmation module
passes pylint10/10 and mypy after an array-conversion/type-annotation cleanup.

Artifacts additionally include `confirmation_report.json` and
`confirmation_predictions.npz`. No new submission or production replacement.
Do not expand to all60,224games merely on the positive single-branch pilot.
The result does not prove extra game coverage is intrinsically useless:
these pilot encoders saw400optimizer players, versus the substantially wider
original encoder training set. However, its actual tested replacement is
not better and should not be presented as a breakthrough.

## Post-replacement additive diagnostic

One additional exploratory check retained original d96, ResNet and opponent
branches and added the fixed cap8 ensemble as a fourth member. No new encoder
training or official-test data was used. The same five player folds fit
nonnegative fusion weights and C1 on fitting players only.

Original0.7106057000 versus additive0.7092566096; delta-0.00134909,
paired-player97.5% interval[-0.00459487,+0.00181089]. Thus retaining the
original branch does not rescue the tested pilot. Fourth-member fitted weights
were0.516–0.584, not percentages; positive fitted weight is not evidence of
improved held-player utility. No further weight sweep or full-scale expansion
is justified by these results. This check was proposed after replacement
results were known and must not be described as preregistered confirmation.

Artifacts: `training/additive_report.json`, `additive_predictions.npz`;
source `coverage_additive.py`, pylint10/10 and mypy passed.
