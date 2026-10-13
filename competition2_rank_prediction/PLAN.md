> SUPERSEDED (2026-10-03 first plan). Current entry point: INDEX.md. Kept for history only.

# Competition 2 (rank prediction) - working plan

Task: 5-20 games of one player -> one of 10 ordered ranks (D C B A 1D..6D).
Score: exact 1.0, adjacent exp(-1)=0.368, else 0. Tutorial baseline (public) 0.4631.
Test sets are released Nov 4 2026; until then only local player-disjoint validation.

## Reused from competition 1 (read-only, via ./c1 symlink; nothing there is modified)
- KataGo binary + NBT model + HumanSL model, KataGoClient, SGF parsing, token builders
- player-disjoint validation, game->player pooling idea

## Rank-specific idea (new)
HumanSL has per-rank policy profiles. The log-likelihood of a player's moves under each
rank profile (rank_11k ... rank_6d) is a direct, calibrated rank signal, plus the
surprisal/entropy features already used in competition 1. Pool per game, then per question.

## Validation
Group split by player_id (8007 players, each in exactly one rank). Questions are simulated
by sampling 5-20 games of one player, scored with the competition metric.

## Stages
1. Pilot: throughput benchmark of KataGo+HumanSL on a few hundred games (GPU 1).
2. Extract features for a player-balanced subset of train games.
3. Per-game/question model (GBM / MLP, ordinal head), group-CV.
4. Optional: official-style ResNet on board planes as ensemble member.
5. After Nov 4: predict public+private, write submission_rank.csv.
