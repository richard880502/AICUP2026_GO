# Identity-conditioned matchmaking residual — P0 audit

Branch: audit/identity-matchmaking-graph

This is deliberately an audit first, not a model.

## Result (2026-11-18): stop at P0-A — no game is stored twice

**The exact graph cannot be built, and P0-B never becomes measurable.** Every
one of the 1,000,000 training rows has a distinct canonical digest:

```text
rows                                 1,000,000
unique_canonical_games               1,000,000
duplicate_hash_groups                        0
two_player_opposite_color_groups             0
accepted_exact_edges                         0
graph coverage, 8,007 labelled players    0.0
graph coverage, 6,807 v47 classes         0.0
```

The audit's own stop condition ("stop immediately if v47 class graph coverage
is small") fires at zero coverage, so no split-half correlation exists to
compare against the 0.15 / 0.30 / 0.40 gates.

### This is the dataset, not a bug

Two checks rule out an extraction artifact.

The digest is player-independent by construction. `open_set_v2/cache.py`
builds `canonical` from the setup properties and the `B`/`W` move coordinates
only — no player id, rank, colour label or metadata enters it, so the same
physical game would hash identically from either side.

A truncation artifact would hide duplicates behind differing digests, so games
were also compared by move prefix:

| Rows sharing this prefix with another row | Count |
| --- | ---: |
| first 40 moves | 392 |
| first 80 moves | 32 |
| first 120 moves | 8 |

If each physical game were stored once per player, roughly 500,000 rows would
share long prefixes. Eight rows at 120 moves is opening coincidence, not paired
storage. The corpus is 1,000,000 distinct games, exactly 100,000 per rank, each
attributed to one labelled player.

### A warning for the inferred-graph version

The obvious salvage is to replace the exact graph with one inferred from the
style encoder, which does identify training-game opponents reasonably often
(0.93-0.95 known share). **The split-half test would then stop being a valid
check.** It controls for repeated games against the same neighbour, but not for
correlated identification error: a player whose opponents are mis-attributed
toward the same hub classes gets the same bias in both halves, so `delta_A` and
`delta_B` would agree for a reason that has nothing to do with matchmaking.
Hubness of exactly this kind is already recorded — it is why CSLS *hurt* in
[TIER_A_RESULTS.md](TIER_A_RESULTS.md), and a related contamination appears in
[SHARED_OPPONENT_AUDIT.md](SHARED_OPPONENT_AUDIT.md), where same-question
opponent pairs average 0.1795 against 0.0296 for cross-question pairs.

An inferred-graph retry would need a different reliability control — for
example splitting by identification confidence, or a permutation null over
attributed neighbours — before any split-half number could be believed.

### Fix applied

`load_rows` rejected the corpus with "encoded row convention assumes <100000"
because rank D has exactly 100,000 rows. Positions run 0..99,999, so a rank
block of exactly 100,000 still fits without colliding with the next rank; the
guard is now `> 100000`. This is the same `rank_index * 100000 + position`
convention `h3_dump.py` already uses on this corpus.

---

Protocol as locked before the run:

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
