# Stage summary: spatial ownership player retrieval

> This document freezes the ArcFace Stage-1 milestone. A later isolated
> group-softmax Top-20 reranker improves the independent proxy from 0.916153
> to 0.923040;
> see `../experiments/reranker/RESULTS.md` for that experiment.

## Decision

The current reproducible checkpoint is **spatial ownership + three-profile
HumanSL + rich profile + query-size ArcFace ensemble**. Its independent proxy
score is **0.916153**.
This is the result to retain as the project baseline; it is not an official
competition leaderboard score.

## Evaluation that the score represents

- 400 black-player candidates, 100 candidate games per player.
- A game-disjoint query cohort for the same 400 candidate identities, evaluated
  with 5, 10, 15, and 20 games.
- The score is the mean of the competition's exponential Top-5 score across
  those four query sizes.
- All encoder, epoch, dimension, and blend choices use only an 80/20 split of
  candidate-library games. Query labels are used only once for reporting.

| Query games | Top-1 | Top-5 | Exponential Top-5 |
| ---: | ---: | ---: | ---: |
| 5 | 0.7625 | 0.9175 | 0.798636 |
| 10 | 0.9000 | 0.9850 | 0.922560 |
| 15 | 0.9550 | 0.9900 | 0.964676 |
| 20 | 0.9700 | 0.9975 | 0.978740 |
| **Mean** | — | — | **0.916153** |

The largest remaining weakness is the 5-game setting. It is also where the
spatial-ownership features produced the clearest improvement.

## Frozen best pipeline

1. Parse black-player SGFs and sample 64 decisions uniformly from each game.
2. Run KataGo NBT at `maxVisits=1`; use decision, policy, ownership, and
   HumanSL `rank_1d`, `rank_5k`, and `rank_5d` signals.
3. Derive ownership-change geometry: 3x3 regions, local share at radii 1/2,
   distance, concentration, entropy, net-change ratio, and direction.
4. Build per-game profiles using phase scales 1, 2, 4, 8, 16, and 32, robust
   distribution statistics, and selected feature interactions.
5. Fit separate 512-d and 1024-d ArcFace encoders for three split seeds.
   L2-normalize and average game embeddings to form player centroids.
6. Retrieve by cosine similarity. Candidate-only validation selects the 1024-d
   ensemble weights 0.70, 0.65, 0.95, and 1.00 for query sizes 5, 10, 15, and
   20, respectively.

`winrate_drop` and `score_lead_drop_scaled` are deliberately omitted: they did
not help the stable profile representation.

## Evidence for the current choice

| Method | Independent proxy | Delta from pre-spatial |
| --- | ---: | ---: |
| Pre-spatial ArcFace ensemble | 0.847215 | — |
| Spatial ownership, global blend | 0.881795 | +0.034580 |
| Spatial ownership, query-size blend | 0.883785 | +0.036570 |
| **+ HumanSL multi-rank, query-size blend** | **0.916153** | **+0.068938** |

Changing the representation was substantially more valuable than changing the
retrieval loss or adding a temporal model. In particular, 64 temporal phase
scales, P×K SupCon, direct classification, local game matching, supervised
UMAP, and diagonal LDA all underperformed the frozen baseline. Exact numbers
and reasons are in `docs/experiment_log.md`.

## Reproduce or inspect

From the project root:

```bash
./scripts/reproduce_best.sh
```

The immutable input profile is
`features/best/humansl_multirank_s64/profiles.npz`; the frozen report is
`results/best/arcface_cross_dimension_ensemble.json`; the visual scaling curve
is `results/best/arcface_cross_dimension_scaling.png`. Raw authorised data is
kept separately at `../data/` and is untouched.

## Recommended stopping point

The multi-rank HumanSL experiment was promoted after beating the prior
checkpoint by +0.032369 under the same candidate-only protocol. The appropriate
next research step for the promoted group-softmax reranker is a fresh outer
cohort or cross-fitted outer evaluation, rather than repeated tuning against
the same 400-player independent proxy.
