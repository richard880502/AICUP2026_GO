# Legal candidate-policy adaptation: stopped at the declared inner gate

Completed staged screen2026-11-09. This is **not** a completed three-seed
comparison and supplies **no new outer utility/Public score**.

At the matched seed0 checkpoint window(epochs5/10):

| Model | Best inner per-game NLL |
| --- | ---: |
| Frozen candidate-policy scores | 2.2946906090 |
| Last-two-block adaptation | 2.2936818600 |

Improvement0.0010087490, below the required0.01. The trainer returned normally
at adapted epoch10and wrote`STOPPED_AT_INNER_GATE`. Seeds1/2and the adapted
outer predictions were not produced. Do not change the threshold, try another
seed, or report a synthetic/inner loss as a competition score.

The frozen control completed40epochs because it is cheap to train on cached
scores, but the cost gate deliberately compares only its epochs5/10, not the
later best checkpoint. Full-score evaluation was refused; the staged report
explicitly marks`full_three_seed_comparison_completed=false`and
`outer_utility_computed=false`.

## Verified work and limits

-All12,000spatial/legal inputs and frozen ten-profile likelihoods completed.
  Legal masks align to game keys, points and validity arrays; no observed
  target move was illegal. Frozen/live initial logits differ by at most
  2.3842e-7; head initialization and initial actual rank loss matched.
-Actual masked-policy gradients update selected backbone parameters while
  the prefix has no gradient. All ten candidate hypotheses are evaluated for
  every game; true ranks never choose metadata or candidate order.
-400optimizer/100inner players are fixed and disjoint from400evaluation
  players. Standardization uses frozen optimizer-game scores only.
-Recorded source/input hashes pass the stopped-run evaluator. Pylint10/10,
  mypy and three unit tests passed before training. GPU1is free after the
  normal stop; GPU0servicewas untouched.

This rejects continued spending on the **specific eight-position, small-cohort,
early-budget recipe**. It does not show that all policy-based fine-tuning is
ineffective, or that an eight-position probe is equivalent to production's
96-decision representation. Its limited exposure must remain explicit.

## Artifacts and decision

`production/policy_adaptation_20261109/`: head/partial-backbone checkpoints,
histories,`inner_gate.json`,`report.json`,source/input hashes and the stop
marker. Frozen control predictions exist but were not scored for outer utility.
Reusable inputs:`policy_frozen_scores_20261109/`and
`policy_legal_inputs_20261109/`under`production/`.

Keep original production C1+C2 and its existing submission. No new CSV,
official-test inference, upload, commit or push. Breakthrough remains unproven.
