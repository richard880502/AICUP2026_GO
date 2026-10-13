# Robust question pooling

This experiment tests whether atypical games should have less influence than
they receive under the production equal-weight geometric mean.  It compares
trimmed, median, central-game, and arithmetic pooling with the production
`logmean` rule.

Selection uses repeated five-fold player-disjoint validation.  The locked
400-player holdout is not read by this experiment.

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.robust_pooling.run
```

## Result

Three repeated five-fold runs selected the existing `logmean` rule:

| Pooling | Mean score |
| --- | ---: |
| **logmean** | **0.63242** |
| trim 10% logmean | 0.63065 |
| central weighted logmean | 0.62550 |
| arithmetic mean | 0.61715 |
| median log | 0.61529 |

Robust outlier removal consistently hurt, so this branch is rejected and the
production pooling rule remains unchanged. The locked holdout was not read.
