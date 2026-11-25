# 1D/2D specialist pilot: no improvement

Completed 2026-11-07 on the previously explored 2,000-player holdout3
cohort, five player folds within each upstream-excluded half. These are
exploratory internal metrics, not official Public or fresh outer results.

| Metric | Frozen C1+C2 | 1D/2D specialist blend |
| --- | ---: | ---: |
| Ten-rank competition utility | 0.717012 | 0.715748 |
| Exact accuracy | 58.20% | 58.00% |
| Error >=2 ranks | 5.10% | 5.10% |
| 1D/2D pair AUC | 0.725725 | 0.701925 (specialist alone) |
| 1D/2D balanced accuracy | 69.25% | 65.50% (specialist alone) |

Paired utility delta is -0.001264, with player-bootstrap 95% interval
[-0.004689, +0.002132]. The two upstream half deltas are +0.000736 and
-0.003264. Specialist AUC is lower than baseline in both halves: 0.6967
vs 0.7253, and 0.7056 vs 0.7255. The ten training-fold pair AUCs range
from 0.733 to 0.806, suggesting a generalization gap for this small pilot.

The fixed label-free router selected 432 of 2,000 questions, including
132 players outside 1D/2D. It changed 36 final decisions: 16 gains and
20 losses; among non-pair players there were five gains and five losses.
Baseline parity and zero-blend parity passed, and pair mass was preserved.

The preregistered stop conditions are met: no utility gain, inconsistent
half directions, and pair AUC below baseline in both halves. Keep C1+C2;
do not expand this recipe to more boundaries or integrate it. This does
not establish that all binary classifiers or additional data are useless.

Each fold trained on 160 pair players (80 per class), and evaluated 200
players across all ten ranks. The 256 inputs concatenate mean per-game
128-D embeddings from the frozen excluded-half own d96 and opponent-specific
seed-0 Transformer checkpoints. The input includes only one seed per branch,
while C1+C2 baseline uses ensembles and other components. This comparison
does not isolate conditional information beyond baseline, and the result
does not rule out a different feature source or a larger safe training set.

Full subgroup, pair-calibration, confusion, route, fold and checkpoint
hash records are available in the local ignored output directory described
in [README.md](README.md).
