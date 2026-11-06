# Competition 2 submission: 2026-10-30 test release

- Official Public score: **0.7227214**, reported by the user on 2026-11-06.
- Private score: not yet disclosed.
- Input: `aicup2026_rank_prediction_tests_20261030.zip`, containing 400 Public
  and 400 Private questions, with 10,000 games in total.
- Method: 96-decision KataGo and HumanSL features, 3-seed player Transformer,
  ResNet snapshots, opponent-specific Transformer (C2), and question-level
  shift residual (C1). All 800 questions used the full C1+C2 method.
- Submission validation: UTF-8, `question_id,pred_rank`, 800 unique question
  IDs, valid rank labels, no missing or extra test IDs, and no fallbacks.
- Local submission: `production/rank_20261030/submission_rank.csv`.
- SHA-256: `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.

The archived run used the local research workspace under
`runtime/competition2_research/`. The relevant extraction, input validation,
prediction, and submission validation source was subsequently synchronized
to `competition2_rank_prediction/`. The local output, models, data and
features are intentionally excluded from Git. KataGo extraction uses random
symmetry, so a fresh extraction is not guaranteed to produce identical CSV
bytes or the same official score. The Public score is not an estimate of the
unseen Private score.
