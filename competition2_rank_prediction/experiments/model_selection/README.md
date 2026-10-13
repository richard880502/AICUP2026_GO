# Production model-set selection

Five-fold player-disjoint question-level fusion consistently assigns almost
all useful weight to the dense Transformer and cosine-schedule ResNet. This
experiment compares that locked two-model set against the historical
four-model set on the 400-player holdout and records paired uncertainty.

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.model_selection.evaluate
```
