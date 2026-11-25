# HumanSL last-two-block adaptation: pilot improves, full pipeline does not

Completed2026-11-09. All results are exploratory on the historically used
400-player validation cohort, not Public or fresh confirmation.

## Matched own-branch comparison

| Three-seed probability ensemble | Official utility |
| --- | ---: |
| Frozen backbone, identical head | 0.3911395106 |
| Adapt final two blocks and trunk norm | 0.3992042846 |

Delta+0.00806477, paired-player95% interval[+0.00338910,+0.01266478].
Seed deltas+0.00767039,+0.00927008,+0.00838410. All three are positive.
Worst rank delta is C -0.01336136. Query-size deltas5–9/10–14/15–20:
+0.01188381/+0.00707254/+0.00567741. The preregistered pilot gate passes.
However, the branch is weak in absolute terms and this is not evidence of
improvement over the full production system.

## Planned complete C1+C2 test

Retain original d96, ResNet and opponent-specific models. Add the selected
adapted ensemble as a fourth member. Fit nonnegative fusion and C1 only on
the other four player folds, preserving all original feature inputs to C1.

| Five-fold player-safe calibration CV | Official utility |
| --- | ---: |
| Original C1+C2, exact replay | 0.7106057000 |
| Original plus adapted HumanSL | 0.7105266850 |

Delta-0.00007902, paired-player97.5% interval[-0.00140932,+0.00129015].
Full-pipeline gate fails. Fitted fourth-member weights vary from approximately
0.000005 to0.1385; they are geometric score coefficients, not percentages.
Positive fitting weight in some folds does not establish utility improvement.

## Verification and limitations

- Six runs completed40epochs×64updates, with matched four-game samples,
  same head initialization and eight positions per game.400optimizer,
  100inner and400evaluation players are disjoint.
- All six selected epoch40 by the minimum of the eight scheduled inner-NLL
  checkpoints. No evaluation utility was calculated during training.
- Whole-cache neutral metadata, representative raw-SGF input replay and
  lossless spatial packing checks passed. Inputs retain legacy handicap
  reconstruction for consistency; this is not a handicap correction study.
- Gradient checks confirmed actual last-block updates and frozen-prefix
  isolation. The vendor tree is clean at pinned commit
  `c560d38585b446bc93b0a62c77364bc8e7d266a4`.
- All32recorded source/input hashes remained unchanged. Six checkpoint
  selection/player-exclusion audits passed; saved own and conditional scores,
  rank deltas and bootstrap intervals replay exactly.
- Original submission SHA256 remains
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
  GPU0service was untouched; GPU1 is free after normal completion.

Small optimizer-player coverage, simple mean pooling/head, eight decisions
and limited update budget constrain conclusions. This rejects the tested
recipe, not every form of pretrained Go-network adaptation. No larger
fine-tuning sweep is justified merely by the positive weak-branch result.

## Artifacts and decision

`production/human_finetune_20261109/`: six partial-backbone/head checkpoints,
histories, validation arrays, source/input hashes, `learning_curves.png`,
`report.json`, `confirmation_report.json`, `confirmation_predictions.npz`.
Inputs remain in `production/human_finetune_inputs_20261109/`.

Keep original production C1+C2. No submission was generated, uploaded or
replaced. The broader breakthrough objective remains unachieved.
