# Hierarchical move experiment

This experiment uses the full one-million-game move-token dataset to train at
the same unit as the competition: a 5-20 game bag from one player.

The first stage extracts a 192-dimensional embedding from the frozen
holdout-safe `move_h.pt` encoder:

```bash
cd /tmp2/richard/rank_prediction
c1/.venv/bin/python -m \
  experiments.hierarchical_move.extract_embeddings --device cuda:0
```

Extraction is resumable with `--resume`. Artifacts are written to
`features/move_embeddings_h/`.

Train the player-disjoint 5-20 game Set Transformer without reading holdout
labels or games:

```bash
c1/.venv/bin/python -m experiments.hierarchical_move.train --device cuda:1
```

If external validation improves, select one blend coefficient on validation
and perform the locked holdout comparison:

```bash
c1/.venv/bin/python -m \
  experiments.hierarchical_move.evaluate_fusion --device cuda:1
```

## Result

Direct bag training improved the move branch from roughly `0.386` to
`0.4297` on the external 400-player validation cohort; its 20-game score was
`0.4719`. This confirms that aligning training with multi-game questions is
useful.

It did not improve the production ensemble. Validation selected an 8% blend,
but the locked holdout score fell from `0.6609` to `0.6397` (paired delta
`-0.0212`, 95% CI `[-0.0398, -0.0026]`). The move hierarchy is therefore a
rejected auxiliary branch, not part of production. The same bag-learning idea
is now being tested on the much stronger dense KataGo/HumanSL representation
under `../dense_player_bags/`.
