# Soft decision-aligned C1 calibration: no confirmed gain

Completed2026-11-09 on the original400-player / 8,000-game calibration-CV
cohort. This is an internal evaluation, not Public.

| C1 residual objective | Official utility | Exact accuracy | Question NLL |
| --- | ---: | ---: | ---: |
| Cross-entropy | 0.7106057000 | 57.2375% | 1.094496 |
| Cross-entropy + 0.2 × soft decoded-action reward loss | 0.7108097151 | 57.2625% | 1.092492 |

Mixed minus NLL utility is `+0.00020402`; paired-player97.5% interval is
`[-0.00101077, +0.00146114]`. The fixed gate required at least `+0.003` and
a positive interval lower bound. It fails. The method improves26 queries and
worsens23.

The soft action is the temperature`0.1` softmax over expected official reward
for each submitted rank. The fixed objective coefficient is`0.2`. The only
change from the NLL control is this C1 loss. Frozen model members, fusion,
features, questions, player-disjoint folds, residual head, optimizer,
regularization and training steps are unchanged. Original baseline utility
replays exactly.

No temperature/coefficient sweep, Public inference, submission, upload,
commit or push was done. Keep the original C1+C2 production submission.
Artifacts are under `production/decision_alignment_20261109/`.

Verification: pylint`10.00/10`; scoped mypy passes all3 Python files;
2 unit tests pass. The production CSV SHA-256 remains
`7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
