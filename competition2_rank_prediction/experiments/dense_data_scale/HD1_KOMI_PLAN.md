# HD1-only final-position komi hypothesis: fixed screen

Written before new engine extraction or score measurement. Motivation and
official-document caveats are in
[the rules audit](../training_overlap/RULES_AUDIT_20261109.md).

## Hypothesis and isolation

HD1no-komi final-position values may provide a more faithful game-result
proxy than the production6.5komi assumption. Do not modify any own/opponent
neural feature, move color, stone placement, encoder or ResNet input. HD>=2
remains untouched in this specific screen, preserving the established model
representation. No claim that Q13's unresolved sections are confirmed.

## Extraction and matched controls

-Use the updated authorized training CSVs to locate exactly1,542HD1games
  belonging to the original8,000validation-game keys. Align SGF moves/color
  with existing cached rows; reject any mismatch or unresolved key.
-For each selected final position, query the existing normal KataGo model
  at64visits, Japanese rules, deterministic symmetry/seed, GPU1only.
-Paired conditions:komi6.5fresh-control andkomi0.5hypothesis. Both rerun in
  the same engine/config. Only winrate and scoreLead change; preserve the
  other four original SGF-derived final-result features.
-Retain original cached6.5as a separate baseline. Fresh6.5controls for
  re-extraction/search variability; compare to both controls, not just old data.
-Require all selected roots to reach64visits and contain finite valid values.
  Store raw Black-perspective outputs, game keys, actor colors and hashes.

## Fixed evaluation

Three complete pipelines:original, fresh6.5HD1values, new0.5HD1values.
All other original members stay frozen. Same five player folds and fixed
queries; C1 and preprocessing fit only on fitting players for each condition.
Original baseline must replay0.7106057000exactly.

Primary contrasts0.5−original and0.5−fresh6.5, paired-player97.5% intervals.
Advance only if overall0.5−original>=0.003and both interval lower bounds>0;
report exposed/unexposed questions, rank/query-size changes and decision
gains/losses. Reject if unexposed utility declines>0.005. No komi grid or
alternate visits search if this fixed test fails. A passed historical-CV
screen still needs confirmation and does not directly authorize submission.

No official-test inputs, manually edited predictions, upload, commit or push.
No new extraction or training job has launched for this hypothesis yet.
