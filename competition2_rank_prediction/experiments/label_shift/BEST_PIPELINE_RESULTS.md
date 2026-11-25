# Uniform-prior adaptation on current C1+C2: reject

Completed2026-11-09 using the current three-member C1+C2 pipeline
(`d96`, `resnet8c`, `opp_scratch`) on the historical 2,000-player
cross-fitted holdout3 cohort. The baseline exactly reproduces the historical
C1+C2 score. Each half was excluded from its upstream model training, and all
fusion/calibration fitting used only the separate400-player validation set.

| Method | Holdout3 official utility |
| --- | ---: |
| NLL-C1+C2 | 0.71701175 |
| Unlabeled per-half uniform-marginal adaptation | 0.71493145 |

Adaptation delta is `-0.00208030`; paired-player95% interval is
`[-0.00732121, +0.00300000]`. Both halves decline (`-0.00289636` and
`-0.00126424`). It changes76of2,000 decisions. Reject this adaptation for
the current best pipeline.

The earlier borrowed-opponent branch scored a positive`+0.003109` diagnostic
on this same cohort. That result does not transfer to the current best model
and must not be used to justify submission changes. The holdout3 labels are
balanced, but the official hidden-test prior remains unconfirmed; even a
correct uniform prior does not guarantee this probability-marginal correction
helps.

No prior or offset sweep, official-test inference, submission, upload, commit
or push was done. The production submission is unchanged. Artifacts are in
`production/decision_alignment_holdout3_20261109/opp_scratch_c{0,1}.npz` and
`production/decision_alignment_best_holdout3_20261109/` plus
`production/label_shift_best_20261109/`.
