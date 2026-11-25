# Ordinal semi-hard mining — initial audit

2026-11-08; issue #1. Design stage, no training or new score yet.

## Hypothesis and controls

Train the existing ten-rank ordinal encoder with an auxiliary player-pair
ranking loss, initially on 1D/2D only. Pairwise gradients must update the
encoder. Preserve per-game inference and the ordinal threshold head.
Compare A (NLL only), B (NLL plus random adjacent pairs), and, conditional
on positive B evidence, C (same loss plus training-only semi-hard mining).
Use common initialization, player splits, sampling and training budgets.
Any C1+C2 comparison requires refitting fusion and residuals safely for
each new model distribution.

## Actual feature inventory

Read directly from cached feature shard metadata before model fitting:

| Source | Shards | Games | Players | Games per player |
| --- | ---: | ---: | ---: | --- |
| features/feat96/train | 16 | 15,211 | 7,607 | 1–2; median 2 |
| dense_player_bags/features | 72 | 36,000 | 2,000 | 18 |

The dense directory contains newly extracted games, not the complete
20-game set: the preceding experiment reports reusing another 4,000 games
from the base features. Those keys and player overlaps must be explicitly
reconciled before claiming 20 usable games per player. The raw training
cache contains zero players with five games; it cannot directly supply
5–20-game auxiliary bags for all 7,607 players.

Base training player counts: D 850, C 777, B 1,008, A 1,542, 1D 393,
2D 570, 3D 578, 4D 600, 5D 618, 6D 671. The new dense directory has 200
players per rank. These are cache inventories, not exclusion-filtered
training counts for a particular checkpoint.

## Before training

1. Reconcile dense game keys with reused base games; audit label agreement,
   unique players, and train/inner-validation/cross-fit exclusions.
2. Check initialization provenance: players seen by a checkpoint cannot
   become its independent validation players during fine-tuning.
3. Write PREREGISTRATION.md after the usable split is established, fixing
   model budget, seed, bag sampling, loss weight, epoch selection and
   conditional mining gate. This plan is not that preregistration.
4. Test player separation, ordinal pairing, differentiable bag scores,
   and zero auxiliary-weight parity before running A/B.

The repeatedly explored holdout3 remains exploratory. Independent outer
confirmation is conditional on positive exploratory evidence.
