# Error analysis and feature visualisation (2026-10-20)

`make_figures.py --holdout-pred holdout_best2_predictions.csv` writes `figures/`:

- `confusion.png` - left: question-level confusion on the untouched 400-player holdout (production best2); right: single-game argmax
  confusion of the dense Transformer on the 8,000 validation games. Row % of each true rank.
- `umap_raw_features.png` - UMAP of per-game mean/std of the 15 dense features (validation games), coloured by rank group.
- `umap_embeddings.png` - UMAP of the dense Transformer's attention-pooled 128-d game vector, per game and averaged per player.
- `umap_by_rank.png` - player-level UMAP, one true rank highlighted per panel.
- `summary.json` - per-rank exact / off-by-one / mean signed error on the holdout.

Findings:
1. Holdout: exact 52.8%, off by one 36.2%, worse 11.0% -> score 0.661. Almost all errors are adjacent ranks.
2. Hardest ranks: 2D (32% exact), A (42%), 3D (45%), 4D (48%); easiest are the ends D (60%) and 6D (75%). Main confusions: A <-> 1D
   (the kyu/dan boundary, 30% of A go to 1D) and 4D <-> 5D.
3. Regression to the middle: low ranks are predicted too high (mean signed error D +0.45, A +0.50, 1D +0.70), high ranks too low
   (4D -0.28, 6D -0.33).
4. Per game, the ordinal head never predicts C or 1D (empty columns): their threshold bands are very narrow. Pooling, the per-rank
   offsets and expected-score decoding recover them at question level (C 52%, 1D 50% exact).
5. The learned representation is essentially one strength axis: player-level vectors lie on a single curve ordered D -> 6D,
   neighbouring ranks overlap along it, there are no separate clusters. Spearman(strength score, rank) at player level = 0.94.
   Raw features show the same gradient with much more overlap.
