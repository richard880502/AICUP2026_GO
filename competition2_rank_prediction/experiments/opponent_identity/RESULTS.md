# Opponent identity as rank evidence (2026-11-14, exploratory)

Submitted by the owner on 2026-11-14: Public 0.7586411 (previous 0.7227214;
holdout3 strict estimate 0.7585). The v4 + v7 version below scored
**0.7599623** on 2026-11-15 and is the final rank submission. Whether this use of the data is acceptable
under the competition rules is the owner's decision (see "Caveat").

## Finding

The opponents in the official rank-test games are mostly labelled training
players. A style encoder trained for player identification
(`competition1_player_identification/experiments/trunk_style`, v4, classes =
6,807 training players) recognises them, which gives the opponent's
*registered* rank instead of an estimate of playing strength.

Share of games whose opponent is a known player, estimated from how often the
top class cosine exceeds the 99th percentile of players known to be absent
(1,200 cohort players v4 never saw), relative to unseen games of players
known to be present:

| Query | Estimated known share |
| --- | ---: |
| Opponents in training games (6,144 sampled) | 0.93-0.95 |
| Opponents in official rank-test games (10,000) | 0.69-0.77 |
| Official rank-test players themselves (10,000 games) | 0.00-0.06 |

Single-game identification of a present player among 6,807: top-1 0.418
(0.460 with move-to-move matching over the 20 best classes), top-5 0.61.
Official test players do not appear as opponents in training games, and
official questions share no game with each other.

## Model

Per game: softmax over class cosines (temperature about 0.04) gives a
posterior over players, hence over the opponent's rank, mixed with a learned
"unknown opponent" share. P(own rank | opponent rank) is a learned function
of the rank difference for three kinds of game (even; HD1 with the opponent
White; HD1 with the opponent Black). Games multiply as evidence with a
learned exponent; the result is added to the log of the existing C1+C2
question probabilities. About 45 parameters, cross-fitted between the two
holdout3 halves.

## Holdout3 (2,000 questions, cross-fitted, expected-score decoding)

"Strict" removes every holdout3 player from the known-player list, which
brings the known-opponent share close to the official one.

| Method | Score | Exact | Paired delta vs C1+C2 [95% CI] | Halves |
| --- | ---: | ---: | --- | --- |
| C1+C2 | 0.7170 | 0.582 | - | - |
| Identity alone, strict | 0.7513 | 0.631 | +0.034 [+0.018, +0.050] | +0.032 / +0.036 |
| Identity + C1+C2, strict | 0.7585 | 0.640 | +0.042 [+0.027, +0.056] | +0.035 / +0.048 |
| Same, weights per question size | 0.7646 | 0.648 | +0.048 [+0.033, +0.062] | +0.041 / +0.054 |
| Identity + C1+C2, all players listed | 0.7746 | 0.662 | +0.058 [+0.042, +0.072] | +0.054 / +0.061 |

Identity alone by number of games (every question truncated, strict):
5: 0.662, 10: 0.736, 15: 0.772, 20: 0.813.

Variants that did not beat the simple model (all within noise of 0.755-0.765):
centroids instead of class directions (0.750), all 8,007 labelled players by
centroids (0.756) or by fitted class directions (0.759), averaging three
encoders (0.761), per-game fallback to the opponent classifier (0.751),
matching-sharpened identification (0.764).

Leak check: v4 trained on some holdout3 players' own moves in the asked
games. Per-game evidence for the true rank is the same for games in and out
of v4's corpus (0.268 versus 0.263 probability on the own rank; 4,429 and
7,623 even games inside mixed questions).

## Follow-up: trained on simulated questions, and game result (2026-11-14)

A question-level model trained only on simulated questions built from the
5,439 training players outside holdout3 (5-14 games each, a random quarter of
the known-player list hidden, players sampled rank-balanced) uses no holdout3
label except the per-size fusion weight:

| Method | Alone | + C1+C2 | Paired delta [95% CI] | Halves |
| --- | ---: | ---: | --- | --- |
| Full 3x10x10 transition tables | 0.7429 | 0.7534 | +0.036 [+0.023, +0.050] | +0.035 / +0.038 |
| Per-game MLP | 0.7414 | 0.7575 | +0.040 [+0.027, +0.054] | +0.039 / +0.042 |
| Tables split by game result | 0.7404 | 0.7567 | +0.040 [+0.026, +0.053] | +0.035 / +0.044 |
| MLP with game result | 0.7345 | 0.7483 | +0.031 [+0.018, +0.045] | +0.026 / +0.036 |

This independently reproduces the gain and the plateau near 0.76. Sampling
players unbalanced costs about 0.03 alone: the remaining training players are
far from rank-balanced (one rank 24%, another 3%) while holdout3 is 10% each.

Game result (HumanSL value head on the asked player's opponent's last three
decisions; 85% of games decided beyond 0.9) adds nothing. Among confidently
identified opponents in even games the asked player wins 45% when registered
one rank below the opponent and 59% when one rank above, so the result barely
separates adjacent ranks.

## Follow-up: rank of stylistic neighbours (2026-11-14, negative)

Each holdout3 player's own games were embedded with the style encoder that
never saw that player (`style_v4_fold{0,1}`), and similarity to the other
half's labelled players (holdout3 removed) gave a rank posterior
(`style_knn.py`, 1,644 questions covered). Alone it scores 0.653 although the
encoder was never trained on rank, but with C1+C2 it is 0.711 against 0.708
for C1+C2 through the same logistic layer (delta -0.001, CI [-0.013, +0.012]).
Playing style carries rank information that C1+C2 already has.

## Official questions

Model fitted on all 2,000 holdout3 questions (strict list), applied with the
full 6,807-class list. 248 of 800 answers differ from the 2026-10-30
submission (15 by two or more ranks; public 65 up / 55 down, private 54 up /
74 down). Mean top probability 0.46 -> 0.66. This file was submitted (Public
0.7586411) despite its name:
`production/rank_opponent_identity_20261114/rank_opponent_identity_NOT_SUBMITTED.csv`.

## Follow-up: v4 + v7 opponent identification (v47): Public 0.7599623

Reported by the user on 2026-11-15; this is the final rank submission.

The opponent side of every game is also embedded with the 960-D style
encoder v7 (`competition1_player_identification/docs/public_v2_trunk_style_20261113.md`),
and the class cosines of v4 and v7 are averaged before the softmax. The two
encoders agree on the top class in only 36% of holdout3 games, so the
average is a better identifier than either. Nothing else changes: the same
~45-parameter model, fitted on the 2,000 holdout3 questions (strict list).

| Method (holdout3 strict, + C1+C2) | Holdout3 | Public |
| --- | ---: | ---: |
| v4 identification | 0.7585 | 0.7586411 |
| **v4 + v7 identification** | **0.7632** | **0.7599623** |
| v4 + v7, weights per question size | 0.7642 | not uploaded |

The holdout3 gain of +0.005 shrank to +0.0013 on Public. Weights per
question size add another +0.001 on holdout3 and were not worth a
submission. Mean top probability of the official answers: 0.45 before
opponent identity, 0.65 after.

## Follow-up: checks for further rank information (2026-11-15, all negative)

These looked for information the leaderboard leaders (Public 0.81) might use
and that the model above does not:

| Check | Result |
| --- | --- |
| Opponents that are Competition 1 pool players | At the threshold passed by 1% of non-pool opponents, 1.9% of test opponents pass (18.6% of original-pool queries do). Almost none. |
| Later games of a question more informative (time drift) | P(opponent rank = own rank) 0.266-0.273 in all five position buckets. Flat. |
| Rank from "who else played this opponent" (nearest training games by opponent style, known or not) | Alone 0.747; on top of C1+C2 + identity -0.007 to +0.002, CI includes 0. |
| Rank-balanced decoding (holdout3 is 10% per rank) | -0.003 / -0.002 on the two halves. The free official predictions are not far from balanced either. |
| Same player split over several official questions | Best-partner similarity of official questions (median 0.501, 9.7% above 0.7) matches Competition 1 candidates known to be distinct players (0.494, 12.7%). No evidence. |
| Official test games inside the 1M training games | None, whether whole games or the first 120 or 80 moves are compared. |
| Test players sharing opponents | Yes: of 1,755 games identified with probability > 0.5, the 982 distinct opponents include 361 met by two or more test players (up to 13). For known opponents this adds nothing beyond their registered rank, which is already used. A joint model over unknown shared opponents was estimated at +0.002 to +0.005 and not built. |

Also negative and recorded elsewhere: self-play KataGo trunk features
(Competition 1 record), ten-profile HumanSL likelihoods and HumanSL policy
fine-tuning (earlier Competition 2 experiments).

Conclusion: opponent identity uses the registered ranks of recognisable
opponents; the remaining error comes from registered ranks that differ from
playing strength and from the ~30% of opponents that are not training
players. No source in the official data closed that gap.

## Caveat

No test label is used; the method is nearest-known-player retrieval with
training labels. It does rely on a property of how the data set was built
(test opponents are training players). `experiments/training_overlap/`
recorded a cautious stance on inferring player identities; confirm with the
organisers' rules before using this for a counted submission.

## Reproduce

The chain that produced the final submission v47 is in this directory.
Run from the repository root with `PYTHONPATH=.:competition1_player_identification`
and `M=competition2_rank_prediction.experiments.opponent_identity`:

1. `python -m $M.map_h3` and `python -m $M.full_qprob`: holdout3 question
   map and the C1+C2 baseline probabilities.
2. `python -m $M.opp_sample 6144` and `python -m $M.link_inputs`: opponent-side
   game lists for training, holdout3 and the official questions.
3. Trunk tokens of those lists with
   `competition1_player_identification.experiments.trunk_style.extract`
   (384-D) and `extract_rich` (960-D).
4. `python -m $M.link_test`: v4 class cosines and controls for the official
   opponents (`test_opp_scores.npz`).
5. `python -m $M.h3_dump`: v4 class cosines for holdout3 (`h3_sims.pt`).
6. `python -m $M.h3_v7`: v7 cosines and the v4 + v7 average.
7. `python -m $M.h3_final_v47`: leak check, fit on holdout3, official answers
   (`rank_opponent_identity_v47.csv`).

Gaps: `h3_opp.csv` was written inline and has no script, and the 384-D token
folders of steps 3-5 (`h3_opp/`, `tokens/`, `present/`, `test_own/`) were
deleted to free disk space, so steps 4-5 cannot be re-run without
re-extracting them. Their outputs are kept (`h3_sims.pt` under
`production/rank_opponent_identity_20261114/`). The intermediate files live
in `/tmp2/richard/claude_scratch/opp_probe/` and
`/mnt/share_data_77/richard/AICUP2026_GO/opp_probe/` on the research machine.

The v4-only scripts and every follow-up probe in this record (simulated
questions, game result, stylistic neighbours, pool overlap, order, kNN,
balanced decoding, duplicate players, exact overlap, shared opponents) are
archived outside Git in `production/rank_opponent_identity_20261114/scripts/`
and `production/research_scripts_20261116/`, not at the repository's lint
standard.
