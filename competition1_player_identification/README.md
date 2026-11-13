# Go player retrieval research

This is the curated source and research record for player retrieval from
anonymous 19x19 SGF records. Large feature caches and model weights are kept
locally and intentionally excluded from Git.

## Current official v2 Public result

**0.7187986**, reported by the user on 2026-11-13: a per-move style encoder on
frozen HumanSL trunk activations answers the new pools
([record](docs/public_v2_trunk_style_20261113.md), `experiments/trunk_style/`).
Earlier the same day: 0.5640548 (two-way fusion) and 0.5525235 (open-set only). See
[the record](docs/public_v2_openset_20261113.md). The paragraphs below
describe the previous best.

**0.5236602**, reported by the user on 2026-11-11. The frozen-encoder
submission improves on the candidate-retrained version's **0.4650701** by
**+0.0585901**. Private performance is unknown. These official mixed-pool
results must not be described using the historical 0.923040 proxy score.

Original public/private predictions are retained. New 600-player pools use
the corresponding original pool's fixed 512D/1024D encoder ensemble and
reranker, retaining source normalization. Only candidate-held-out games
select dimension blends and reranker strength; no neural weights are updated
on new pools. Candidate-selection scores are not independent test scores.

The current upload artifact remains local at
`../production/player_v2_frozen_20261111/submission_player.csv`.
The inferior upload CSV was removed at the user's request; its feature cache
and original-pool source models remain because the current method reuses them.
See [the official record](docs/public_v2_20261111.md) and
[reproduction instructions](production_pipeline/README.md).

## Current best offline result

The [new mixed-color 600-way experiment](experiments/open_set_v2/README.md)
is complete. Its locked, player-disjoint final score is **0.295403**, versus
**0.057627** for the fixed Fisher-profile baseline, on ten references and
1/2/3/5-game queries. These are different-protocol research scores, not Public
or the historical proxy below. See [complete results and figures](experiments/open_set_v2/RESULTS.md).

**0.923040** mean exponential Top-5 score on the independent 400-way proxy.
This is an offline research result, not a competition leaderboard score. It
adds a group-softmax Top-20 hard-negative reranker to the reproducible
**0.916153** ArcFace baseline. Reranker fitting, early stopping, and its blend
weight use only
candidate-library synthetic queries; independent labels are not model inputs.

| Query games | Exponential Top-5 score |
| ---: | ---: |
| 5 | 0.810298 |
| 10 | 0.935521 |
| 15 | 0.964225 |
| 20 | 0.982115 |

The frozen ArcFace stage is a **+0.068938** absolute gain over the pre-spatial
baseline; group-softmax reranking adds another **+0.006887** on the independent
proxy.
The candidate-only selection uses a query-size-specific 512-d/1024-d ArcFace
blend, followed by cosine retrieval of L2-normalized player centroids.

## Best method

- 64 uniformly sampled black-player decisions per game, evaluated with
  KataGo NBT at `maxVisits=1`.
- Base decision signals plus HumanSL `rank_1d`, `rank_5k`, and `rank_5d`
  policy surprisal, rank, and entropy.
- Ownership-map spatial change: local share at radii 1/2, change distance,
  concentration, entropy, net-change ratio, direction, and 3x3 regions.
- Rich per-game profiles: phase scales 1, 2, 4, 8, 16, and 32; mean/std,
  robust distribution statistics, and feature interactions.
- `winrate_drop` and `score_lead_drop_scaled` are excluded.
- Three seeds are evaluated separately for 512-d and 1024-d ArcFace encoders;
  candidate-only validation selects an independent blend for each query size.

## Layout

| Path | Role |
| --- | --- |
| `features/best/humansl_multirank_s64/` | Local untracked decision cache, profile, and schema for the best result. |
| `results/best/` | Best result JSON and its rendered scaling curve. |
| `experiments/reranker/` | Isolated reranker cache, code, results, and interpretation. |
| `docs/` | Reproduction guide and concise research log. |
| `archive/` | Prior baselines, rejected implementations, and pilot-only artifacts. |
| `go_retrieval/` | SGF parsing, deterministic split, KataGo client, and result helpers. |
| `models/katago/` | Local untracked NBT, HumanSL, and retained TF3 weights. |
| `.venv/` | Local untracked Python environment. |

The authorised raw training data remains outside this directory at
`../data/`; this cleanup never deletes or moves it.

## Reproduce the best result

After activating `.venv`, run:

```bash
./scripts/reproduce_best.sh
```

It retrains the candidate-only ensembles and writes a fresh report under
`results/best/`. The exact frozen Stage-1 report is
`results/best/arcface_cross_dimension_ensemble.json`. The isolated reranker
commands and current-best report are in `experiments/reranker/README.md` and
`experiments/reranker/RESULTS.md`. After the embedding cache exists, reproduce
the current reranker with:

```bash
.venv/bin/python -m experiments.reranker.run_group_mlp --device cuda:1
```

## Stage conclusion

Spatial ownership gives the first large gain; adding HumanSL behavior at three
rank profiles raises the independent proxy from 0.883785 to 0.916153, and the
group-softmax distributional reranker raises it to 0.923040. The prior
mainlines are
preserved under `archive/`. See
[docs/stage_summary.md](docs/stage_summary.md) for the current handoff and
[docs/experiment_log.md](docs/experiment_log.md) for the full decision record.
