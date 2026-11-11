# Frozen original-pool encoders for v2, 2026-11-11

Upload artifact:
`production/player_v2_frozen_20261111/submission_player.csv`.
Validation: PASS, all 1,600 questions and five distinct in-pool players each.
The user reports official **Public 0.5236602**, versus **0.4650701** for the
previous submission (delta **+0.0585901**), on 2026-11-11. Private is unknown.
These scores average original and new Public questions; they are not the
historical independent-proxy result of 0.923040.

The original public/private 400-question predictions are copied exactly.
For each new 600-player pool, the corresponding original pool's three-seed
512D/1024D encoders and group-softmax reranker are reused without neural
training. Their original normalization is retained; the classification heads
are not used for retrieval. Only new candidate-held-out games select dimension
blends and frozen reranker strength. Actual query answers are unavailable.

Same initial candidate-held-out queries, same five reference games per player,
same query sizes 1/2/3/5, and Stage-1 exponential Top-5 metric:

| New pool | Candidate-trained encoder | Frozen original-pool encoder |
| --- | ---: | ---: |
| Public | 0.0583158 | 0.1586105 |
| Private | 0.0607467 | 0.1547619 |

Adding the frozen source reranker, with strength selected on these same
candidate queries, gives 0.1627247 and 0.1620859. These are model-selection
diagnostics, not independent or leaderboard results. They should not be
compared as a paired independent gain with the earlier reranker reports that
used a different player subset and repeated query sampling. No confidence
interval or predicted official gain is claimed.

Artifacts in the new production folder include `frozen_report.json`,
`comparison.json`, `submission_validation.json`, per-pool `selection.json`,
and the execution log. The comparison records source-checkpoint SHA-256
digests, both submission digests, and changes by pool. The prior, inferior
upload CSV was deleted at the user's request; its recorded SHA-256 is
`c2bf315f0085eb43cf1c432749e18939a328338b8baad25325e8e8d503f79249`.
Cached features and original-pool checkpoints remain available for the frozen
method. The current CSV SHA-256 is
`2b206a72a4b77c7367c1008ee9a330c11bfea5ecfc364755a61b30450e3e97ac`.

The changed Python module passed pylint 10/10 and mypy. Final validation
re-read the saved CSV, checked its digest against the validation report,
and verified that original public/private predictions did not change.
