# Player identification v2: per-move HumanSL trunk style encoder

Score reported by the user on 2026-11-13. Private is unknown; no official
query answers were used for fitting or selection.

| Submission | Official Public |
| --- | ---: |
| Frozen original-pool encoders (2026-11-11) | 0.5236602 |
| Open-set v2 recipe on new pools | 0.5525235 |
| Two-way fusion on all four pools | 0.5640548 |
| **Trunk style encoder on new pools, original pools unchanged** | **0.7187986** |

Gain over the previous best: **+0.1547438** Public. Only the 800 new-pool
answers changed, so the new Public pool improved by about +0.31.

## Method (`experiments/trunk_style/`)

The earlier encoders read one 9,198-D statistical summary per game, built
from 41 engine output scalars at 64 sampled decisions. This encoder reads
the decisions themselves:

- `extract.py`: for the same 64 evenly spaced own decisions, the frozen
  pretrained HumanSL network (b18c384, raw weights, neutral 1d metadata)
  is run on the pre-move position. A token is the 384-D trunk activation at
  the played point, the 384-D global trunk mean and a pass flag (769-D,
  stored fp16). No identity or rank label enters the network.
- `train.py`: each token is concatenated with the 41 engine scalars of the
  same decision (810-D), standardized on training games, and a four-layer
  256-wide Transformer with attention pooling outputs a 256-D unit vector
  per game. Loss: ArcFace (s=30, m=0.2) plus supervised contrastive
  (tau=0.1) on 128 players x 4 games per batch, 15% decision dropout.
  Training players only: the 6,807 open-set v2 training players, at most 16
  games each (the existing 120,561-game engine corpus). The 600-player
  validation cohort selects the epoch (18 of 40).
- `apply.py`: centroid cosine retrieval, fused with the locked open-set
  score at the validation-selected weight 3.0.

## Offline results

| Final cohort, query games | Open-set v2 | Style encoder alone | Top-1 |
| ---: | ---: | ---: | ---: |
| 1 | 0.169 | 0.384 | 33.0% |
| 2 | 0.266 | 0.590 | 53.5% |
| 3 | 0.329 | 0.704 | 66.0% |
| 5 | 0.418 | 0.816 | 77.3% |
| mean | 0.2954 | **0.6234** [0.597, 0.650] | |

Validation 0.6235. Fused with open-set: validation 0.6327, final 0.6244;
paired delta versus open-set on final +0.3290 [+0.3044, +0.3547]. The final
cohort was not used for training, epoch selection or the fusion weight.

Transfer check on official data, using candidate player IDs only (one
reference game as the query against the other nine):

| Validation | Final | pub2 | priv2 |
| ---: | ---: | ---: | ---: |
| 0.344 | 0.338 | 0.335 | 0.354 |

The earlier engine_512 encoder scores 0.091 on the same pub2 check.

Ablation (same data, 24 epochs, engine scalars zeroed): best validation
about 0.605, so the 41 scalars add roughly 0.02 on top of the trunk tokens.

## Limits

- The original/new split of Public is not observable; with the original
  pools near 0.88 the new Public pool would now be about 0.55, below the
  offline 0.62. That decomposition is an assumption.
- Validation peaks at epoch 18 and then declines: the encoder overfits the
  16-game-per-player corpus, so more games per player are the next step.
- Trunk tokens for 120,561 games take 12 GB; extraction runs at about 21
  games/s per RTX 3090.

Artifacts: `production/trunk_style_20261113/` (`style_v1/checkpoint.pt`,
`submission_style_v1/`). Upload SHA-256
`e5af2704ec3c446c46bc1ab47db6ec4ee193e8f57487f0bf783e84940b22edbf`.
