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

## Follow-up the same day: more training games, all four pools, Public 0.8084515

| Submission | Official Public |
| --- | ---: |
| Style encoder v1 on new pools (above) | 0.7187986 |
| **Style encoder v2 on all four pools** | **0.8084515** |

Gain **+0.0896529** Public, reported by the user on 2026-11-13. Upload
SHA-256 `aea54a41b6e0f3f55788bf09c9647d4b6ba356f2de879a8d89d0723591230052`,
artifacts in `production/trunk_style_20261113/submission_style_v2/`.

### Encoder v2 (`train.py --extra --engine-dropout 0.3 --epochs 30`)

`inputs.py` adds up to 32 further games per training player (138,684 games;
many players have fewer than 48 games in total), so training grows from
about 109k to 248k games of the same 6,807 players. Only trunk tokens were
extracted for the added games. Their 41 engine channels are zero, and the
engine channels of the other games are dropped with probability 0.3 during
training so the encoder works with or without them. Engine scalars are
still used at inference, where every game has them.

| | v1 | v2 |
| --- | ---: | ---: |
| Validation (epoch selection) | 0.6235 | 0.6683 |
| Final cohort, alone | 0.6234 | 0.6818 [0.658, 0.707] |
| Final cohort, fused with open-set (weight 3) | 0.6244 | 0.6877 |

Final cohort by query games (v2 alone): 1: 0.452, 2: 0.659, 3: 0.745,
5: 0.870. Adding games without engine scalars improved the encoder, so
re-running KataGo for them was not needed.

### Original pools (`apply.py`, `original_pool`)

The same encoder, which never saw any pool player, is applied to the
100-game libraries. Its centroid cosine is added to the pool-trained
Stage-1 score over the full pool; one weight for all query sizes is chosen
on candidate-held-out games (20 per player, 8 draws per size). The Top-20
group-softmax reranker is no longer applied, because it was fitted to the
pool-trained ordering.

| Held-out candidate games, mean over 5/10/15/20 | Public pool | Private pool |
| --- | ---: | ---: |
| Pool-trained Stage 1 | 0.9259 | 0.9107 |
| With style encoder (selected weight 24) | 0.9939 | 0.9932 |
| Style encoder alone | 0.9935 | 0.9931 |

Official questions whose Top-1 changed: 76/400 (public), 32/400 (private).

### Reading the official score

The original/new split of Public is still not observable. The two halves
now sum to 1.617; if the original pools score near the held-out estimate
(0.97-0.99), the new Public pool is about 0.63-0.65, slightly below the
offline 0.69. Remaining headroom is therefore almost entirely in the new
pools.

## Every own move instead of 64 samples: Public 0.8345466

| Submission | Official Public |
| --- | ---: |
| Style encoder v2, 64 decisions, all four pools | 0.8084515 |
| **Interim all-moves encoder (v4a) on new pools** | **0.8345466** |

Gain **+0.0260951** Public, reported by the user on 2026-11-13. Only the
new-pool answers changed; original pools keep the v2 answers. Upload SHA-256
`a27c5a047d7a3caeaf09e69dbb14f80eca720af4296d99a155c1bfb433af1331`,
artifacts in `production/trunk_style_20261113/submission_style_v4a/`.

### Why the token count was tested first

Two 20-epoch ablations of the v2 recipe (248k games, 64 decisions):

| Variant | Best validation |
| --- | ---: |
| v2 as trained | 0.6683 |
| every second decision only (32 tokens) | 0.5753 |
| global trunk mean zeroed | 0.6572 |

Halving the decisions costs 0.09, so more decisions per game were expected
to help; the global mean is worth about 0.01 and doubles storage, so it was
dropped from the new layout.

A third run added the official candidate libraries as extra training players
(2,000 players, a fifth of each player's games held out). Validation stayed
at 0.667 and the held-out candidate games moved within noise, so that
variant was not used.

### All-moves layout (`extract.py --samples 160 --local-only`, `train.py --all-moves`)

- Every own non-pass move is a token, up to 160 per game (99% of games have
  at most 160; longer games are sampled evenly). Mean 114 tokens per game.
- A token is the 384-D played-point trunk activation only.
- The 41 engine scalars exist for the 64 originally sampled decisions. They
  are placed at those decisions' positions (`scatter_engine`) with a presence
  flag; other tokens carry zeros. Placement was verified against the earlier
  tokens: cosine 1.000 at the computed positions, 0.65 when shifted by one.
- Encoder input 426-D, otherwise the same model and losses.

Interim encoder v4a: the 16-game corpus only (109k training games), 30
epochs, no engine dropout.

| | v1: 64 decisions, 109k games | v2: 64 decisions, 248k games | v4a: all moves, 109k games |
| --- | ---: | ---: | ---: |
| Validation | 0.6235 | 0.6683 | 0.7122 |
| Final cohort, alone | 0.6234 | 0.6818 | 0.7270 [0.704, 0.750] |

Final cohort by query games (v4a): 1: 0.491, 2: 0.707, 3: 0.813, 5: 0.897.
Fusing with the open-set score no longer helps (final 0.7239, validation
0.7073 versus 0.7122 alone); the uploaded file still used the fused score.
`apply.py` now uses the style score alone when validation favours it.

The official gain on the new Public pool (+0.052) matches the offline final
gain (+0.045). A full run with the 139k extra training games in the same
layout was still extracting when this was written.
