# C1 official-reward calibration objective: no confirmed gain

Completed 2026-11-09 on the original 400-player / 8,000-game calibration-CV
cohort. This is an internal comparison, not Public.

| C1 residual objective | Official utility | Exact accuracy | Question NLL |
| --- | ---: | ---: | ---: |
| Cross-entropy | 0.7106057000 | 57.2375% | 1.094496 |
| Cross-entropy + 0.2 × (1 − expected official reward) | 0.7106847151 | 57.2500% | 1.091735 |

Mixed minus NLL utility is `+0.00007902`; paired-player 97.5% interval is
`[-0.00128022, +0.00149132]`. The fixed advancement gate required at least
`+0.003` and a positive lower interval bound, so it fails. The mixed objective
slightly lowers question NLL but does not produce a meaningful score gain.
It improves 28 queries and worsens 27.

The screen uses the exact same frozen `d96`, `resnet8c` and `opp_scratch`
members, five player-disjoint outer folds, fusion fitting, C1 feature inputs,
question samples, residual architecture, initialization, Adam settings,
regularization and 300 training steps. Only the fixed C1 residual objective
changes. The original pipeline score replays exactly.

The objective coefficient was fixed at `0.2` before the outer score was
computed. No coefficient search, new submission, Public inference, upload,
commit or push was done. Keep the original C1+C2 submission. Artifacts are
under `production/reward_calibration_20261109/`.

Verification: pylint `10.00/10`; scoped mypy passes all 3 Python files;
2 unit tests pass. Original production CSV SHA-256 remains
`7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.
