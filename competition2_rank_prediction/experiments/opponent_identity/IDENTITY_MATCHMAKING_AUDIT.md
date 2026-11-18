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


---

# Follow-up: cross-encoder identity residual audit

The exact graph failed because the corpus does not store both sides of a
physical game. The remaining hypothesis is narrower:

> same-rank opponent identities may carry persistent registered-rank residual
> information, but it must survive validation by a *different* identity
> encoder.

This follow-up stays on the same branch but uses a different reliability
control from the failed exact-graph split.

## Why cross-encoder

A single style encoder can create false stability through hubness: the same
classes may attract assignment error in both halves. Therefore the audit does
not accept v4->v4 or v7->v7 self-consistency as evidence.

Instead:

```text
v4 posterior on source-player half A
    -> estimate per-class delta_j
    -> validate class alignment + held-out residual with v7 on half B

v7 posterior on source-player half A
    -> estimate per-class delta_j
    -> validate with v4 on half B
```

The direction is reversed and both source-player halves are used, giving four
independent direction/fold checks.

## Split rule

The 6,144 sampled training games are split by **source player_id**, not by
game. Every game of one labelled source player stays in the same half.

This prevents one player's persistent own-rank behaviour from leaking into
both estimation and validation halves.

Only HD0/HD1 games enter the primary audit because the official test contains
no HD>=2 games.

## Soft identity assignment

No top-1 hard identity is used. Each game contributes the complete class
posterior:

```text
q_gj = softmax(cosine_gj / 0.04)
```

with the source player's own v47 class masked when applicable.

For each encoder and training half:

1. fit the low-capacity rank-only transition
   `E[own rank | opponent registered rank, game kind]`;
2. compute each game's residual around that transition;
3. posterior-weight the residual into each opponent identity;
4. shrink class residuals toward zero with denominator `mass + 2.0`.

This produces `delta_j` in units of registered-rank residual.

## Reliability control: within-rank permutation null

For a class to count at a mass threshold it must have enough posterior mass in
both the estimator and validator halves.

Primary mass threshold: **2.0 effective games**.

For every cross-encoder class-level correlation, opponent identities are
randomly shuffled **within the same registered rank** 1,000 times. The
observed identity alignment therefore has to beat a null that preserves:

- registered rank;
- number of classes per rank;
- residual scale by rank.

This is specifically designed to reject a result that is merely rank
information restated through identity labels.

## Held-out predictive check

On the untouched source-player half, the validator encoder first makes the
rank-only opponent prediction. The estimator encoder's frozen `delta_j` is
then transported through the validator posterior:

```text
correction_g = sum_j q_validator(g,j) * delta_estimator(j)
```

The report measures:

- correlation of this correction with the validator's held-out rank residual;
- MAE before / after;
- rounded contest-like utility before / after;
- posterior mass landing on reliable identities.

No scalar correction weight is tuned on the evaluation half; the residual is
applied in its natural rank units.

## Official coverage

The official v4+v7 opponent posterior is read only to answer:

> if the residual were real, how much of the official opponent posterior lands
> on identities with enough training evidence?

No official labels are used.

## Predeclared gate

At the primary mass threshold 2.0, proceed only if all are true:

- at least 50 reliable identities in every direction/fold check;
- mean of Pearson/Spearman over the four checks >= **0.25**;
- at least **3 of 4** checks beat the within-rank permutation null at p<0.05
  for both Pearson and Spearman;
- mean held-out rounded-utility delta >= **+0.003**;
- no direction/fold has negative correlation or negative utility delta;
- official mean v47 posterior mass on reliable identities >= **0.25** when
  official caches are available.

Failing the gate closes the entire identity-specific matchmaking-residual
route. Passing it justifies one small frozen-delta v47 experiment, not a GNN.

## Run

```bash
git fetch origin
git checkout audit/identity-matchmaking-graph

GPU=1 \
bash competition2_rank_prediction/experiments/opponent_identity/\
run_cross_encoder_residual_audit.sh
```

The runner reuses token folders if present. Otherwise it extracts v4 local
HumanSL tokens and v7 rich tokens for the existing 6,144-game
`opp_sample.csv`.

Primary output:

```text
runtime/competition2_research/experiments/
  identity_matchmaking_graph/cross_encoder_report.json
```
