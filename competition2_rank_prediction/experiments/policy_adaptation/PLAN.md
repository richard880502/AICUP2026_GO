# Legal candidate-policy adaptation: staged controlled screen

Fixed before fitting rank classifiers. This follows policy/gradient feasibility
and four-game engine parity, not an observed rank-score improvement.

## Data and representation

-Reuse the fixed500-player coverage manifest:400optimizer/100inner players,
  eight games each. Original400evaluation players/8,000games stay disjoint.
-Each game supplies eight own positions. Every position is scored under all
  ten candidate profiles, irrespective of its actual rank. Japanese simple-ko
  legal support excludes all suicide and is independent of candidate rank.
-Per-game features:ten mean legal played-move log probabilities. No engine
  visit/search features, hidden-feature classifier or temporal Transformer.
-True labels select training CE targets only. They never choose metadata,
  candidate profiles, position samples, masks or evaluation-time routing.
-Keep the baseline-compatible legacy handicap reconstruction; no inference
  patch or claim of corrected handicap modeling.

## Matched frozen and adapted models

-Same110-parameter Linear10→10head, initialized identically for each seed.
-Fit per-channel feature mean/std on frozen optimizer-game scores only and
  use the same transform in both arms. Inner/evaluation labels never fit it.
-A uses cached frozen scores. B uses the same initial checkpoint and updates
  only blocks16/17and final trunk norm. Policy head/metadata encoder stay
  frozen, backbone.eval()preserves all running normalization statistics.
-Headlr0.001, backbonelr0.00001, AdamWweight_decay0.01, gradient clip1,
  fixed40epochs×64updates, batch4games. Cosine schedule spans all2,560steps.
  Matched player-balanced/game draws; seeds0/1/2. No hyperparameter sweep.
-Checkpoint candidates5/10/15/20/25/30/35/40:earliest minimum inner per-game
  NLL. Same800inner games, full ten-profile computation in both arms.

## Staged cost gate

Ten-profile gradients and evaluation are substantially more expensive than
the preceding hidden-feature pilot. Before committing all six fits, seed0
has a fixed inner-only gate at epoch10:adapted best inner NLL from epochs5/10
must be at least0.01below frozen best NLL from epochs5/10. Otherwise stop
this recipe and report an incomplete/stopped staged screen, not a completed
three-seed result. Do not score outer utility or try another seed to rescue it.
If it passes, continue the same optimizer/schedule/sampling state through40
epochs and complete all three matched seeds. Do not reset the schedule.

## Primary scoring after stage passage

Three-seed probability-ensemble whole-question official utility B−A, historical
fixed queries, paired-player95% bootstrap. Report all seeds, rank and query-size
effects. Advance to original C1+C2 plus B only for delta>=0.005, lower bound>0,
2/3positive seeds and worst-rank loss>=-0.05. Full conditional calibration must
fit only on other player folds and replay original baseline. This historical
cohort is exploratory; a pilot gain never authorizes production adoption.

## Runtime preparation

All12,000legal caches are complete and aligned to spatial inputs, about2.4MiB
compressed. Selected moves are legal; bit packing is exact. A real cached-input
GPU check confirms finite nonzero last-block gradients after legal normalization,
with no frozen-prefix gradient. Frozen-score caching has launched in session95127,
writing only `production/policy_frozen_scores_20261109/`. That handle is historical
context and must be polled to establish live state. No rank fitting has launched.
GPU1only; GPU0serviceoutofscope. No automatic submission/commit/push.
