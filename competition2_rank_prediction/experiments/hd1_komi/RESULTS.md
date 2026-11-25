# HD1-only final-value komi hypothesis: no confirmed gain

Completed2026-11-09. Exploratory original400-player calibration CV, not
Public or a fresh confirmation cohort. The Q13official-document caveats
remain unresolved; this was a controlled hypothesis, not a production patch.

| Complete C1+C2 condition | Official utility |
| --- | ---: |
| Original cached6.5komi | 0.7106057000 |
| Fresh6.5komi on HD1only | 0.7106057000 |
| New0.5komi on HD1only | 0.7110927452 |

0.5−original:+0.00048705, paired-player97.5% interval
[-0.00109326,+0.00209326]. 0.5−fresh6.5:+0.00048705, interval
[-0.00102720,+0.00207034]. The primary minimum+0.003and positive-interval
requirements fail. No komi-grid or visits sweep is scheduled.

Among8,000fixed questions,5,134contain at least one HD1game. Their mean
delta is-0.00040066; unexposed questions improve+0.00207722. C1refitting can
change unexposed predictions too. Thus the small aggregate improvement does
not support the intended exposed-game correction mechanism.42questions gain,
35lose. A/1Dpositive subgroup deltas are descriptive, not new selection gates.

## Isolation and verification

-Exactly1,542selected updated-source HD1games; all8,000original validation
  keys resolve. Selected moves/actor IDs/ranks/colors match old cached rows.
-3,084paired roots all reach exactly64visits. Fresh6.5and0.5queries differ
  only in request ID and komi. Mean Black-perspective score shift is+6.00935
  points, consistent with the intended input change; not proof of true rules.
-Only actor-perspective final winrate/scoreLead are replaced. All other
  four facts and nonselected game rows stay exact. No colors or stones change.
-All neural members remain fixed. C1and its preprocessing refit separately
  using only the other320players in each fold. Original baseline replay passes.
-Recorded training-input/source hashes, root budgets and saved utility/rank/
  interval replay pass. Pylint10/10 and mypy pass for four source files;
  three tests cover paired-query isolation, actor conversion/fact preservation,
  and rejection of under-budget roots.

Artifacts:`production/hd1_komi_20261109/paired_roots.npz`,
`predictions.npz`,`report.json`,`manifest.json`and completion marker.

Decision:keep original C1+C2. No official-test inference, new submission,
upload, commit or push. Original submission hash is unchanged; GPU1is free.
This does not establish that6.5is the true HD1komi or rule out benefits from
a differently trained, fully rule-correct model. It rejects this isolated
final-proxy replacement as a demonstrated score improvement.
