# Ten-profile policy and legal-support equivalence, 2026-11-09

The unmodified raw PyTorch checkpoint was compared to the official engine
on four fixed training games×eight own decisions×ten candidate profiles.
This is32distinct positions with320profile-conditioned policy comparisons,
not320independent positions or a scored rank experiment.

After applying engine legal support and renormalizing:

-Mean total variation0.0005123973; maximum0.0019261679.
-Maximum absolute policy difference0.0019234325.
-All320×362legal-support entries match independent Python replay exactly.
-Played-move log-probability absolute error:mean0.0011325222nats,
  maximum0.0065274260nats.

The preexisting HumanSL-equivalence thresholds(meanTV<=0.005,maxTV<=0.02)
pass. No thresholds were loosened after measurement. Python legality uses
simple ko plus explicit exclusion of all suicide moves under Japanese rules;
pass is included. This does not establish correctness under other rule sets,
superko or corrected handicap reconstruction.

`likelihood.py` retains gradients through legal masking and log-softmax.
Direct comparison on all320cases matches independently normalized legal
played probabilities within1e-6. Two tests verify zero gradient/mass from
an illegal high-logit move and rejection of illegal/out-of-range targets.
Pylint10/10 and mypy passed for all five experiment modules.

Artifacts:`production/policy_equivalence_20261109/report.json` and
`policies.npz`, containing raw policy, engine policy and independent legal
masks. No adaptation was performed in this equivalence run. The preceding
gradient/memory probe used raw policies and must not be represented as a
benchmark of the masked implementation. A real pilot must use legal masks,
fixed candidate ordering and player-disjoint frozen/adapted controls.

No new rank score, official-test inference or production replacement exists.
