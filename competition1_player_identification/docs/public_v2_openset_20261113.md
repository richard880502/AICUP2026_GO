# Player identification v2: open-set recipe on the official new pools

Score reported by the user on 2026-11-13. Private is unknown; no test labels
were used for fitting or selection.

| Submission | Official Public |
| --- | ---: |
| Candidate-retrained ArcFace and group-softmax | 0.4650701 |
| Frozen original-pool encoders and reranker | 0.5236602 |
| **Original pools unchanged + locked open-set v2 recipe on new pools** | **0.5525235** |

Gain over the previous best: **+0.0288633** Public. Only the 800 new-pool
answers (`pub2_`, `priv2_`) changed, so the new Public pool improved by about
**+0.058** (Public averages the original and new 400 questions equally).

## What was done

`production_pipeline/open_set_official.py` applies the sealed
`experiments/open_set_v2` configuration (`locked_configuration.json`) to the
official 600-player pools, reusing the cached 64x41 decisions and 9,198-D
profiles under `production/player_v2_20261111/{pub2,priv2}`. Nothing is
fitted: encoders, normalization, pair reranker, rank prior and fusion weights
are loaded as saved. Questions are padded into the experiment's
[600, 10 references + 5 queries] layout and only the slice for each
question's real game count is read.

`--check` replays the sealed final cohort through the same code path:
0.295410 versus the recorded 0.295403 (3 of 2,400,000 rank positions differ,
bf16 batching).

## Offline score did not fully transfer

The pre-upload estimate was about 0.60 Public, assuming the new pools would
score near the offline 0.295. The label-free confidence check (top1-top2
margin distributions matching the offline cohort) was not a reliable
predictor. The original/new split of Public is not observable. The three
uploads are consistent with, for example, original about 0.88 and new pool
about 0.05 -> 0.17 -> 0.22, but this decomposition is an assumption, not a
measurement. Query colour mix and candidate colour balance match the offline
cohort; official new pools contain no HD>=2 games and fewer HD1 games
(24% versus 32%).

## Artifacts

`production/player_v2_openset_20261113/submission_player.csv`, SHA-256
`c4f8bc22851498177b87a4499ff7b04e05cfa89ff1ee8c47a7ddb836a6a0d6cc`; per-pool
`predictions.csv` and per-source `scores.npz`; `submission_validation.json`.

## Follow-up the same day: two-way fusion, Public 0.5640548

| Submission | Official Public |
| --- | ---: |
| Open-set recipe on new pools (above) | 0.5525235 |
| **Both method families fused on all four pools** | **0.5640548** |

Gain **+0.0115313** Public, reported by the user on 2026-11-13. Produced by
`production_pipeline/fusion_official.py`; report in
`production/player_v2_fusion_20261113/fusion_report.json`, upload SHA-256
`b079cdf0113e8d400c5ba552b01c55f8d34b1b234cc055e0ee9fabbf3ab78b5c`.

- New pools: the private original pool's six ArcFace encoders are added to
  the locked open-set fusion with weight 0.3, selected on the open-set
  validation cohort (0.2894 -> 0.2960). Final cohort, read once:
  0.2954 -> 0.2994, paired delta +0.0040, 95% CI [-0.0040, +0.0120].
- Original pools: the open-set engine_512/1024 centroid score is added to
  the pool-trained Stage-1 score before the unchanged Top-20 reranker.
  Weights per query size (0.2-0.6) are selected on candidate-held-out games:
  mean Stage-1 gain about +0.004 (public pool) and +0.009 (private pool).
  With weight 0 the code reproduces the previous answers on 400/400
  questions in both pools.

The offline estimate was about +0.004 Public; the observed +0.0115 is larger.
With 800 Public questions this difference is within sampling noise, so no
claim is made about which half contributed the gain.
