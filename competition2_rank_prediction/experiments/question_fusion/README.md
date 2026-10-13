# Question-level fusion

This low-capacity experiment learns four nonnegative model weights and ten
class biases directly from simulated 5-20 game questions. Selection uses
stratified, player-disjoint cross-validation.

Run cross-validation first:

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.question_fusion.run
```

Holdout evaluation is deliberately opt-in and should only be used after the
cross-validation result justifies it:

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m experiments.question_fusion.run --holdout-only
```
