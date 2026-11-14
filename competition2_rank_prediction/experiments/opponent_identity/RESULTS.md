# Opponent identity as rank evidence (2026-11-14, exploratory)

Not adopted and not submitted. Whether this use of the data is acceptable
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
74 down). Mean top probability 0.46 -> 0.66. Candidate file (not submitted):
`production/rank_opponent_identity_20261114/rank_opponent_identity_NOT_SUBMITTED.csv`.

## Caveat

No test label is used; the method is nearest-known-player retrieval with
training labels. It does rely on a property of how the data set was built
(test opponents are training players). `experiments/training_overlap/`
recorded a cautious stance on inferring player identities; confirm with the
organisers' rules before using this for a counted submission.

## Reproduce

Scripts and intermediate files are kept outside Git in
`production/rank_opponent_identity_20261114/` (`scripts/`: `link_inputs.py`,
`link_test.py`, `h3_link.py` -> `h3_dump.py`, `h3_bayes.py`, `h3_bayes_b.py`,
`h3_final.py`, `h3_leak.py`, `h3_trunc.py`, `gallery8007.py`, `cls8007.py`,
`ens_id.py`, `id_match.py`). They have not been brought to the repository's
lint standard.
