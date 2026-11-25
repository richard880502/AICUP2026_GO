# Reward alignment results — 2026-11-08

## Conclusion

**The small reward term helps the matched retrained control, but has not
established superiority over original C1+C2. Keep the Public-best original
0.7227214 as production.** No alternate submission was generated or uploaded.

GPU1 completed six fits: seeds0/1/2 × NLL/mixed, 40epochs each. Original
own-side ordinal Transformer architecture; mixed loss = NLL +0.2 ×
(1−expected official reward). ResNet and opponent C2 stayed frozen. The
experiment replaced d96, rather than adding Joint or another fusion member.

## Same-cohort full C1+C2 comparison

400 original validation players, 8,000 fixed questions, five player/rank
stratified folds for fitting fusion and C1. No outer players entered
gradient training, normalization or checkpoint selection. Question counts
range5–20, exactly matching the established baseline evaluation procedure.

| Own branch and checkpoint selection | Utility | Exact | Question NLL |
| --- | ---: | ---: | ---: |
| Original frozen C1+C2 | 0.710606 | 57.2375% | 1.094496 |
| Retrained NLL / inner NLL | 0.708275 | 56.9125% | 1.096764 |
| Retrained NLL / inner question utility | 0.711001 | 57.3000% | 1.096975 |
| NLL+0.2reward / inner question utility | **0.712950** | **57.6375%** | 1.099430 |

The baseline reproduces the previous Joint evaluation baseline exactly:
0.7106057000201695. These are NOT Public scores and not the original
2,000-player cross-fit0.717012. No new official score is available.

## Attribution and uncertainty

Average repeated query differences within each player, then bootstrap400
players5,000 times. The two planned contrasts against matched NLL/NLL
use97.5% intervals (Bonferroni for two comparisons).

| Comparison | Utility delta | Player97.5% interval |
| --- | ---: | --- |
| Utility selection vs matched NLL selection | +0.002725 | [-0.000182,+0.005787] |
| Mixed+utility selection vs matched NLL selection | +0.004675 | [+0.001396,+0.008055] |

The mixed arm passes the fixed **matched-control screening gate**; all
three seed diagnostics improve (+0.003036,+0.001701,+0.000493), and no rank
declines against that control. Seed diagnostics use ensemble-fitted
calibration, not three independently fitted production pipelines.

Additional exploratory attribution, same97.5% intervals (not a new claim
of familywise control across all supplementary comparisons):

- Mixed+utility vs **original production**: **+0.002345**,
  interval **[-0.001207,+0.005943]** — superiority remains unconfirmed.
- NLL+utility vs original: +0.000395, interval[-0.002802,+0.003598].
- Mixed+utility vs NLL+utility: +0.001949,
  interval[+0.000129,+0.003740] — suggests reward itself adds a small lift,
  not solely the checkpoint-selection change.

Do not advertise+0.004675 as an improvement over the old best: the matched
control is weaker than the original. Its training-only normalization and
dense inner NLL selection differ from the historical trainer. The frozen
original comparison is the relevant production comparison.

## Findings

1. Minimizing NLL and maximizing official utility select different epochs.
   NLL-arm selections: NLL28/34/37 versus utility12/25/21; mixed-arm utility
   selects9/25/21. Later NLL improvements do not guarantee utility gains.
2. Utility/exact improve while outer question NLL worsens slightly.
   This is consistent with the calibration tradeoff anticipated before
   training; it is not evidence that the probability estimates improved.
3. Multi-game selection required actual multi-game inner data. Original
   inner players have approximately two games each. Their held-out dense
   games were supplemented, never added to optimizer rows;196/228/190
   inner players per seed qualify for genuine5/10/15/20-game questions.
4. The historical outer validation cohort has been studied repeatedly.
   Fold isolation prevents direct calibration leakage but does not undo
   adaptive research reuse. A fresh-player replication, excluding those
   players from every trained upstream branch, is the next confirmation
   step. Do not expand lambda search based on this outer result or Public.

## Verification and artifacts

- `audit.json`: all three player partitions disjoint; all6 histories have
  exactly40epochs; selected epochs equal earliest inner-metric optima.
- `test_objective`:6tests pass, including all100 entries of the official
  reward table, exact loss/gradient equivalence at lambda0, finite gradient,
  invalid inputs, official-size query generation and inner-only supplement.
- `pylint competition2_rank_prediction/experiments/reward_alignment`:
  **10.00/10**, no suppressions.
- `mypy --follow-imports=silent --ignore-missing-imports` on the same path:
  **7source files pass**. Imported legacy code/missing dependency stubs
  are intentionally outside this scoped check.
- Original CSV SHA-256 unchanged:
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
- Ignored local output: `production/reward_alignment_20261108/` contains
  checkpoints, source-hash manifest, splits, histories, audit,
  `evaluation.npz`, `report.json`, `production_comparison.json`,
  `learning_curves.png`, `comparison.png`.
- The incomplete sparse setup is preserved separately and not evaluated.
  Training and evaluation are terminal; GPU0 vLLM was never touched.

See [reproduction commands](README.md) and the [fixed protocol](PLAN.md).
