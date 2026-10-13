# Go player retrieval research

This is the curated source and research record for black-player retrieval from
anonymous 19x19 SGF records. Large feature caches and model weights are kept
locally and intentionally excluded from Git.

## Current best offline result

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
