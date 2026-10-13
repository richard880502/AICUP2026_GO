# Distribution-aware bag ranker

This experiment learns from complete 5-20 game questions instead of treating
each game as an independent rank-prediction task.

It uses the cached probabilities of the frozen `d96_h`, `move_h`,
`resnet5_h`, and `resnet8c_h` models. Model choice and the blend with the
existing pipeline are selected using player-disjoint five-fold validation.
The 400-player holdout is evaluated only after those choices are locked.

Run from this directory:

```bash
../../c1/.venv/bin/python run.py
```

Outputs are written under `results/` and do not replace the production
pipeline unless the holdout result is better and statistically defensible.
