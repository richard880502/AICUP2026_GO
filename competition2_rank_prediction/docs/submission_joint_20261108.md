# Upload-ready alternate: C1+C2 plus joint model, 2026-11-08

Local CSV:
`production/rank_joint_20261108/submission_rank_joint_20261108.csv`

SHA-256:
`46e4dfeba0b403083c3c5ca92c65b7bd660c233b8379f63636c87048c44e1e2a`

This is a separate experimental submission prepared at the user's request.
Official Public/Private performance is unknown; no upload was performed.
The previously reported Public0.7227214 belongs to the original submission.

## Method and rationale

The first attempt added the three mining checkpoints to C1+C2. On400
original validation players excluded from mining training, geometric
fusion assigned the new member weights of only about1e-5–4e-5. Five-fold
utility was identical and all800 test decisions matched the baseline.
The delivery guard stopped that identical candidate.

The fallback uses the earlier joint_game primary method: one Transformer
consumes96 own plus96 opponent decision tokens with side/time indicators.
Three new models (seeds0/1/2,40 epochs, batch256) were trained using the
original training feature pool and player-disjoint inner epoch selection.
The400 external validation players are absent from training. C1's original
own/opponent/final features are retained; the new joint member changes
geometric fusion before the validation-refitted C1 shift correction.

## Internal evidence

Five folds, stratified by player/rank. Fit fusion and C1 only on320 players,
evaluate80, with20 shared5–20-game queries per player. Average repeated
query changes within player before5,000 bootstrap resamples.

| Method | Calibration-CV utility | Exact |
| --- | ---: | ---: |
| C1+C2 baseline | 0.710606 | 57.24% |
| C1+C2+joint | 0.711988 | 57.41% |

Delta+0.001382,95% player interval[-0.001718,+0.004556]. This is exploratory
evidence on a previously studied validation cohort. It does not establish
superiority or predict a particular Public/Private score. The earlier
2,000-player cross-fit joint result0.7218 is a different model/cohort and
is not relabeled as this submission's score.

Final fusion weights (not constrained to sum to one):
d96=0.176876, resnet8c=0.285415, opp_scratch=0.044346, joint=0.859761.
The weights and C1 use the full validation cohort for official inference.

## Delivery audit

- Original authorized2026-10-30 test ZIP,400 Public+400 Private questions,
  10,000 games; existing own/opponent/final feature caches reused.
- Strict validator PASS: UTF-8, exact `question_id,pred_rank` header,
  800 unique IDs, all expected questions and valid ranks, original order,
  no missing/extra/empty entries. No fallback predictions.
- Exactly31 decisions differ from the original CSV; all changes are one
  rank. No class-balanced decoding or Public-based threshold selection.
- The baseline was refitted and replayed exactly, including CSV bytes.
  Original SHA-256 remains
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
- New per-rank counts: D89,C94,B87,A84,1D73,2D76,3D75,4D78,5D81,6D63.
- Checkpoints and source/input provenance are documented in the local
  manifest and [reproduction guide](../experiments/mining_submission/README.md).
  Per-question methods, probabilities, CV and full validation output are
  saved alongside the new CSV.

The stable inference default and original submission have not been replaced.

## Official Public feedback, 2026-11-08

User-reported Joint Public score: **0.7195608**, versus original C1+C2
**0.7227214**, a difference of **-0.0031606**. Reject this alternate as a
production replacement. The small exploratory internal lift did not
transfer to Public. Private remains unknown. Preserve both artifacts and
do not tune this submission against repeated Public feedback.
