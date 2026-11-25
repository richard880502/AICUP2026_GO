# Conditional HumanSL output context — rejected 2026-11-09

## Result

No demonstrated complement to the complete frozen C1+C2 pipeline.
All8,000validation games were extracted successfully, with usable normal
and all three HumanSL streams for every game. Missing-game counts are zero.

Five rank-stratified player-disjoint calibration folds,400players,
20fixed5–20-game questions per player. Fit fusion, shift correction,
standardization and any imputation only on fitting-player games/questions.

| Method | Exploratory calibration-CV utility |
| --- | ---: |
| Original C1+C2, original lambda0.01 | 0.710606 |
| C1+C2, matched lambda0.1 control | 0.707827 |
| Add normal-KataGo context, lambda0.1 | 0.708992 |
| Add normal+HumanSL context, lambda0.1 | **0.711007** |

Primary paired-player97.5% intervals, after averaging repeated questions
within each of400players before5,000resamples:

- Human−original: **+0.000401**, interval **[−0.011283,+0.012167]**.
- Human−normal: +0.002014, interval[−0.004402,+0.008899].

Both intervals cross zero. Gain is below the preregistered0.005threshold;
`screen_pass=false`. No tuning of phases, profiles, regularization or
feature subsets after seeing the result. No new submission created.
Original Public-best0.7227214 remains unchanged; these are not Public scores.

## Interpretation

HumanSL's value/uncertainty outputs genuinely vary across profiles, but
availability and variation alone do not imply conditional rank information.
This particular16-decision, before/after, mean/std representation does not
establish useful extra signal over production. Broad intervals also warn
against treating the tiny mean lift as reliable. This does not prove that
all spatial HumanSL representations or hidden activations are unhelpful.

The original validation cohort has been used historically, so even a
positive result would have required fresh-player confirmation. The
regularization/normal-output controls prevent claiming all differences
as a human-profile effect. Do not retune this branch on the official Public.

## Verification

- Exact original baseline replay:0.7106057000201695.
-8shards ×1,000games,8,000unique game keys; exact alignment of player IDs,
  ranks and games with original cached validation predictions.
- Values finite; pre-move winrates in[0,1]; uncertainty fields nonnegative.
  Explicit unit tests for BLACK-to-player perspective and actual post-move
  alignment. Invalid responses are masked rather than filled as valid zeros.
- Extractor SHA matches extraction-settings manifest. Model/engine
  provenance and raw official field references are in[README.md](README.md).
- Independent calculation from saved predictions matches all four official
  utilities. Predicted distributions finite/normalized;400unique players
  and8,000queries covered.
- `pylint competition2_rank_prediction/experiments/human_context`:10.00/10.
- `mypy --follow-imports=silent --ignore-missing-imports` on same target:
 5files pass; imported legacy code/missing dependency stubs outside scope.
-6unit tests pass; `git diff --check` passes; no warning suppressions.
- Original submission SHA unchanged:
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
- GPU1 extraction/evaluation terminal; GPU0 vLLM untouched.

Local artifacts: `production/human_context_20261109/` contains raw features,
settings, `evaluation.npz`, `report.json`, `comparison.png`.
Source[plan](PLAN.md),[commands](README.md), and all output retained.
The broader goal of finding a breakthrough remains open.
