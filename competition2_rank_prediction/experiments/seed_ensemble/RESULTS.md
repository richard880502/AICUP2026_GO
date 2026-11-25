# Opponent-model seed ablation: keep all three seeds

Completed2026-11-09 on the historical 2,000-player cross-fitted holdout3
cohort. The all-three-seed C1+C2 baseline exactly reproduces0.71701175.
Per-seed validation and holdout predictions averaged back to their existing
cached probabilities within1e-6. Fold exclusion lists matched all scored
players.

| Opponent seeds kept | Utility | Delta vs all3 | Bonferroni98.333% player CI |
| --- | ---: | ---: | ---: |
| 0,1,2 | 0.71701175 | — | — |
| 1,2 | 0.71748327 | +0.00047152 | [-0.00448978, +0.00544255] |
| 0,2 | 0.71529933 | -0.00171242 | [-0.00750515, +0.00399437] |
| 0,1 | 0.71651175 | -0.00050000 | [-0.00550515, +0.00455696] |

No leave-one-out combination meets the preregistered advancement gate.
Seed1 removal is particularly harmful in half0; seed2 removal is slightly
negative in both halves. Keep the full three-seed mean. This does not prove
that every future seed ensemble will behave the same way.

The 12 individual checkpoint passes (two model folds × validation/holdout3 ×
three seeds) ran on physical GPU1; GPU0's vLLM service was untouched. No
training, Public inference, submission, upload, commit or push occurred.
Artifacts are under `production/seed_ensemble_20261109/`.

Verification: pylint10.00/10; scoped mypy passes all3 Python files. The
three-seed mean replay and player-exclusion checks pass in the evaluator.
