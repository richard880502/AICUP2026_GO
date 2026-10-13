# Value and ownership player-bag experiment

This experiment augments the current 20-game player bag with normal-KataGo
decision features that are absent from the strongest 414D / 606D branches:

- win-rate and score-lead loss caused by the human move;
- total and spatial ownership change;
- phase, location, and ownership-change geometry.

The existing value extraction covered only roughly two games per training
player, so it could not be fairly evaluated in the 20-game bag architecture.
This directory contains a resumable 32-decision extraction for all 40,000
selected games. It uses GPU 0 only and does not interact with the separate
GPU 1 workload.

Run or resume:

```bash
cd /tmp2/richard/rank_prediction
NUM_WORKERS=8 DEVICE=0 bash experiments/value_player_bags/run_extraction.sh
```

After all 72 feature shards exist, build the aligned embeddings, train the
value-augmented Transformer, and score it on external validation only:

```bash
DEVICE=cuda:0 bash experiments/value_player_bags/run_training.sh
```

## Result

The branch gave a small repeatable lift when blended into the frozen
cross-feature system on two validation samplings, but a validation-locked
weight of 0.12 decreased the CPU holdout score from 0.677955 to 0.675856.
It is retained as a documented negative result, not included in the best
submission pipeline. See `BOTTLENECK.md` and
`results/transformer_full/holdout_cross_value_locked_cpu.json`.
