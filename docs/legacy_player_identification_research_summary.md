# Player Identification research summary

## Final verified scores

All numbers below use the same fixed local 400-player proxy and the
competition's exponential Top-5 metric. They are offline results, not Public
leaderboard scores.

The final comparison chart is at
`best_64step_player_retrieval/results/plots/final_method_comparison.png`; the
group-softmax training curve is at
`best_64step_player_retrieval/results/plots/group_softmax_training.png`.

| Method | Mean score | Delta from previous |
|---|---:|---:|
| Official tutorial ResNet + triplet | 0.035958 | — |
| Rich profile + ArcFace 512D/1024D ensemble | 0.916153 | +0.880195 |
| Frozen ArcFace + binary distributional reranker | 0.917371 | +0.001218 |
| Frozen ArcFace + group-softmax MLP reranker | **0.923040** | **+0.005669** |

| Query games | Official tutorial | ArcFace Stage-1 | Reranker best |
|---:|---:|---:|---:|
| 5 | 0.015728 | 0.798636 | 0.810298 |
| 10 | 0.028477 | 0.922560 | 0.935521 |
| 15 | 0.041809 | 0.964676 | 0.964225 |
| 20 | 0.057820 | 0.978740 | 0.982115 |

The new best is `0.9230399459`. The official tutorial was reproduced for all
75,000 training steps using all 100 candidate games per player and reached
`0.0359584373` on the identical cohort.

A second, player-disjoint 400-player official cohort was also trained for the
full 75,000 steps. Its fixed mean was `0.0256586313`; across 50 query resamples,
the mean of the four query-size scores was `0.0245127627`. The second cohort's
fixed 5/10/15/20-game scores were `0.022153`, `0.016081`, `0.031053`, and
`0.033348`. This confirms cohort sensitivity, but both cohorts remain far
below the rich-profile system. The saved checkpoint, all 48,000 per-game
embeddings, and resampling report are under
`aicup_tutorial_review/results/multisplit_seed20260920/`.

## What improved the score

The largest improvement comes from representing player decisions explicitly:

- KataGo NBT policy and ownership signals
- HumanSL policy behavior at multiple human ranks
- ownership-change spatial geometry
- robust multi-scale whole-game distributions
- 512D/1024D ArcFace ensembles selected only on candidate-library validation

The isolated reranker keeps that Stage-1 frozen, retrieves Top-20 hard
candidates, and adds 1024D game-to-game similarity distribution features. A
group-softmax MLP scores all 20 candidates jointly. Candidate-only,
player-disjoint validation selected epoch 18 and neural weight 1.0. Stage-1
was reproduced exactly at `0.9161533072`, and reranking reached
`0.9230399459` on a single independent evaluation.

## Interpretation

The official recent-8-move board-window encoder learns a real but weak player
signal: its score grows from 0.015728 with 5 games to 0.057820 with 20 games.
However, it omits the policy naturalness, ownership response, and long-range
distributional behavior that dominate this task. It is useful as an end-to-end
baseline, but the experiment gives no reason to replace the rich-profile
mainline with it.

The group-aware reranker gain is positive on both candidate-only validation
and the independent cohort, with the largest gains at 5 and 10 query games.
The next defensible experiment is a fresh outer cohort or cross-fitted outer
split. Repeated tuning against the current independent cohort should be
avoided.

## Authoritative artifacts

- Official reproduction result:
  `aicup_tutorial_review/results/internal_proxy_demo.json`
- Official second-cohort multisplit result:
  `aicup_tutorial_review/results/multisplit_seed20260920/report.json`
- Official comparison and protocol:
  `aicup_tutorial_review/OFFICIAL_TUTORIAL_COMPARISON.md`
- Original ArcFace best:
  `best_64step_player_retrieval/results/best/arcface_cross_dimension_ensemble.json`
- Reranker current best:
  `best_64step_player_retrieval/experiments/reranker/results/group_softmax_mlp.json`
- Reranker code and detailed analysis:
  `best_64step_player_retrieval/experiments/reranker/`
