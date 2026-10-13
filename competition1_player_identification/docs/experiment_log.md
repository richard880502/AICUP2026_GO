# Experiment log

## Evaluation protocol

All values below use a 400-player black-only proxy. A deterministic seed picks
the candidate cohort; a second deterministic cohort contains disjoint games
for the same candidate identities. Model
selection uses candidate games only. The independent query result is reported
after selection and is not an official competition submission score.

The scalar is the mean, over query sizes 5, 10, 15, and 20, of the competition
Top-5 exponential score: ranks 1 through 5 receive 1, `exp(-1)`, `exp(-2)`,
`exp(-3)`, and `exp(-4)` respectively.

## Accepted baseline

| Method | Independent proxy | Decision |
| --- | ---: | --- |
| Spatial ownership + multi-rank HumanSL + query-size ArcFace blend | **0.916153** | Current best |
| Spatial ownership + query-size ArcFace blend | 0.883785 | Superseded; preserved in archive |
| Spatial ownership + global 1024-d ArcFace | 0.881795 | Superseded; preserved in archive |
| Pre-spatial 512-d + 1024-d ArcFace ensemble | 0.847215 | Superseded; preserved in archive |
| Pre-spatial 1024-d ArcFace ensemble | 0.845297 | Previous useful component |
| Pre-spatial ArcFace + Gaussian fusion | 0.846548 | Rejected: lower than baseline |

The multi-rank profile's candidate-only validation selected the 1024-d blend
weights 0.70, 0.65, 0.95, and 1.00 for 5, 10, 15, and 20 query games.
Full per-query metrics are stored in
`results/best/arcface_cross_dimension_ensemble.json`.

| Query games | Pre-spatial | Spatial, global blend | Current | Current - global |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.657663 | 0.736321 | 0.798636 | +0.062315 |
| 10 | 0.843956 | 0.881362 | 0.922560 | +0.041198 |
| 15 | 0.936672 | 0.938212 | 0.964676 | +0.026464 |
| 20 | 0.950568 | 0.971286 | 0.978740 | +0.007454 |

Spatial ownership gives the large gain, especially at five query games. The
query-size blend adds a smaller candidate-selected gain, chiefly at 15 games.

## Rejected directions

| Direction | Independent proxy | Finding |
| --- | ---: | --- |
| 64 temporal phase scales | 0.822548 | Finer segmentation hurt generalization. |
| P×K supervised contrastive loss | 0.702559 | Much weaker than ArcFace for this profile representation. |
| Direct 400-class classifier | 0.6137 | Does not match retrieval geometry. |
| Local game matching | 0.389 | Per-game nearest neighbours are too noisy. |
| Supervised UMAP | 0.035 | Collapsed retrieval quality. |
| Shared-covariance diagonal LDA | 0.570869 | Worse than player-specific metric learning. |

## Engine observations

- The TF3 `kata1-tf3-b11c768-s11001M-d5973M` pilot is very highly correlated
  with the NBT decision signals, so switching networks alone is unlikely to
  yield a large gain.
- KataGo ownership is valid for this experiment. It is a board-wide value map,
  and comparing pre/post move maps yields a player decision signature.
- The analysis client now drains KataGo stdout concurrently. Without this,
  large batches can deadlock when the output pipe fills.
- Setting `analysisPVLen=1` removes unnecessary long principal variations;
  all required policy, root, and ownership fields remain available.

## Spatial-ownership result

The full 400 × (100 candidate + 20 query) cache was extracted with the
dual-3090 KataGo configuration, then evaluated using exactly the same
candidate-only ArcFace protocol as the previous baseline. It adds the
following fields at no extra engine-evaluation cost beyond the ownership maps
already requested:

- local change share at radii 1 and 2;
- distance, concentration, and entropy of change;
- signed net-to-absolute ownership-change ratio;
- direction of change relative to the played coordinate;
- existing 3×3 regional change summary.

With a global blend, the resulting independent proxy was **0.881795**, an
absolute gain of **0.034580** over 0.847215. A candidate-only,
query-size-specific blend raised it to **0.883785**. That spatial-only cache is
now retained as a superseded baseline under `archive/baseline_0883785/`.

The one-off extraction-and-merge script and its unmerged engine cache are in
`archive/provenance/`. They are retained for audit, while the active path uses
the already merged best cache and `scripts/reproduce_best.sh`.

## Multi-rank HumanSL result

HumanSL policy surprisal, rank, and entropy at `rank_1d` and `rank_5d` were
extracted for the exact same 400-player, 100-candidate, 20-query, 64-decision
cohort; the existing `rank_5k` fields remain. The merged profile scored
**0.916153**, an absolute gain of **+0.032369** over spatial ownership alone.
All four query sizes improved, with the largest gain at five games. The merged
cache is now the active best at `features/best/humansl_multirank_s64/`; raw
HumanSL extraction remains under `archive/provenance/` for audit.
