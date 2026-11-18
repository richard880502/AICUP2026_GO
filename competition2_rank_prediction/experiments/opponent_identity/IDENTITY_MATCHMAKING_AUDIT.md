# Identity-conditioned matchmaking residual — P0 audit

Branch: audit/identity-matchmaking-graph

This is deliberately an audit first, not a model.

The question is: among players with the same registered rank, does opponent identity carry a persistent matchmaking-position offset that is stable across different neighbours?

If not, do not build the proposed identity-conditioned v47 or a graph model.

## P0-A — exact game-pair graph

The Competition 2 raw cache already stores a canonical SHA256 digest built from setup stones plus the complete move sequence.

If one physical game appears once for each labelled player, the rows should share the exact canonical digest while having two distinct player_id values and opposite colours. Such a group creates an exact player-player edge. No style encoder or identity inference is involved.

The audit reports training rows, unique canonical games, duplicate hash groups, exact two-player/opposite-colour groups, unique player-player edges, all-player graph coverage, v47 6,807-class graph coverage, and degree statistics.

Accepted edges are written to:

runtime/competition2_research/experiments/identity_matchmaking_graph/exact_edges.csv

## P0-B — persistent identity residual

For player j define a simple matchmaking-neighbour statistic: mean registered rank of DISTINCT exact-graph neighbours.

Players of different registered rank obviously have different neighbour ranks, so that rank effect is removed:

delta_j = mean(neighbour rank of j) - mean neighbour rank among players with registered_rank(j)

Each player's distinct neighbours, not games, are deterministically split into two disjoint halves. The audit independently computes delta_j_A and delta_j_B and reports Pearson/Spearman stability for players with at least 1, 2, 3, or 5 neighbours in EACH half.

This prevents repeated games against the same opponent from manufacturing a stable-looking offset.

## Optional official sanity check

If test_opp_scores.npz and controls.npz are present under /tmp2/richard/claude_scratch/opp_probe, the report also shows how often the official v4 top-1 opponent class lands on a graph-covered v47 player, including the confidently identified subset.

This is only a cheap P0 coverage check. It is not the final v4+v7 posterior-mass calculation.

## Run

From repository root:

git fetch origin
git checkout audit/identity-matchmaking-graph
bash competition2_rank_prediction/experiments/opponent_identity/run_identity_matchmaking_audit.sh

CPU only. No HumanSL/KataGo extraction and no neural training.

Main output:

runtime/competition2_research/experiments/identity_matchmaking_graph/report.json

## Decision gates

Stop immediately if v47 class graph coverage is small or split-half residual correlation is approximately zero.

If players with at least 3 neighbours in each half have Pearson/Spearman below roughly 0.15, the hypothesis is weak.

Proceed to the posterior-weighted identity-offset experiment if split-half Pearson or Spearman is roughly at least 0.30 with non-trivial v47-class and official-opponent coverage.

A split-half correlation around 0.4+ with broad v47 coverage is a strong signal and would justify the next stage:

P(opponent identity | game) × identity-specific delta_j -> fractional matchmaking rank -> v47 rank transition.

Only after that small model shows cross-fitted utility gain should we consider Node2Vec, matrix factorisation, or GNN-style graph embeddings.

## Why this differs from previous negative work

The earlier nearest-training-game probe was game-level: opponent embedding -> similar training games -> rank.

This audit tests a persistent player-level latent variable: opponent identity j -> many distinct matchmaking neighbours -> stable delta_j.

The experiment is intentionally easy to falsify before adding model capacity.
