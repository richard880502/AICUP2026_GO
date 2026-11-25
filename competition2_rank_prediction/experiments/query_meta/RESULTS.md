# Nonlinear query-level C1+C2 residual: reject

Completed2026-11-09 on the historical 2,000-player cross-fitted holdout3
cohort. The baseline exactly replays0.71701175. The candidate replaces only
the final question-level calibration with a small zero-initialized residual
MLP trained on nested player-disjoint OOF queries.

| Method | Official utility | Exact accuracy |
| --- | ---: | ---: |
| Current NLL C1+C2 | 0.71701175 | 58.20% |
| OOF-trained nonlinear query residual | 0.69462844 | 54.95% |

Candidate minus baseline is `-0.02238332`; paired-player95% interval is
`[-0.03388859, -0.01037244]`. Both outer folds decline (`-0.02164241` and
`-0.02312422`). It loses about2.24score points and is rejected.

The MLP's epoch was selected by held-player OOF NLL (epoch50, NLL1.04228),
then refit on all OOF validation queries. This NLL-selected checkpoint did
not transfer to the discrete expected-reward metric. The experiment provides
no support for using this nonlinear residual; no shrinkage or epoch sweep was
run against holdout3.

The baseline's three members, data, player-disjoint outer models and
expected-reward decoder are unchanged. Ten inner calibration fits were made
from the two upstream model folds × five player folds. Holdout3 labels were
used only for final evaluation. No official-test inference or submission was
created. The production submission is unchanged.

Verification: pylint10.00/10; scoped mypy passes all4 Python files; one
identity-initialization unit test passes. GPU1 was used for the small MLP;
GPU0 vLLM remained untouched. Artifacts are under
`production/query_meta_20261109/`.
