# Training/test overlap diagnostic

Run `python -m competition2_rank_prediction.experiments.training_overlap.audit`
from the repository root, with the research environment activated.

Result: the one-million-game training index has zero same-actor and zero
opposite-actor move-sequence matches against all 800 official rank questions
(10,000 games). Output: `production/training_overlap_20261109/report.json`.

The legacy 64-bit move hash ignores root metadata. A positive-control check
matched 16 known training games with their correct actor colors. Unit tests
cover header/whitespace invariance, same/opposite actors and absent matches.
No training rank labels or other-task candidate libraries were loaded by the
audit, and no predictions were changed.

**Zero repeated games does not establish player-disjointness.** No identity
lookup approach should be inferred valid from this result. Organizer policy
on train/test player overlap remains unconfirmed.

Checks: pylint 10/10; mypy passed for three files; two unittest tests passed.
