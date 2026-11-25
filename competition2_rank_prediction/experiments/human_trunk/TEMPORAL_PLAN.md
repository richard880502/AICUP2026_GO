# Full-feature temporal probe — separate follow-up 2026-11-09

The first PCA64/linear pilot scored0.711679 versus0.710606, delta0.001073
with97.5% CI[-0.000856,+0.002999]; it failed its gate and is not adopted.
This follow-up is explicitly registered AFTER that negative result. It is
not portrayed as a predeclared successful continuation of the linear pilot.

Question: does linear PCA/phase compression discard useful interactions
within the now-verified frozen769Ddecision representation?

Fixed before this follow-up's outcomes:

- Reuse the exact existing train15,211/validation8,000games and frozen
  HumanSL raw-weight embeddings. No new extraction, profiles or labels.
- No PCA. Feed normalized16×769decision tokens into the SAME ordinal
  RankTransformer used by original production: d128,4heads,3layers,FFN384,
  dropout0.1. Trunk remains frozen; temporal probe trained from scratch.
- Seeds0/1/2;40epochs, batch256, AdamW lr0.0005, weight_decay0.05,
  original OneCycle schedule, token_dropout0.15, Gaussian noise0.05.
- NLL loss; minimum per-game NLL on original10%player-disjoint inner split
  chooses earliest checkpoint. Do not combine reward/utility selection,
  mining, soft labels or new hyperparameter searches with this feature test.
- Channel normalization fit ONLY on each seed's optimizer rows; inner and
  original validation excluded. Explicitly report train/inner/outer overlap0.
- Average the three models' validation probabilities. Add as a fourth
  member to original frozen C1+C2; retain original own/opponent C1 features.
  Same fivefold player-stratified calibration protocol, original baseline
  replay, original advancement gate gain>=0.005,97.5%CI lower>0,
  no rank decline>0.05. No Public-based decisions or automatic submission.
- This is exploratory on a previously studied cohort, not fresh confirmation.
  Report alongside the failed linear probe, not just whichever looks better.

Output: production/human_trunk_temporal_20261109; same experiment source
directory. Original production and linear-pilot artifacts never overwritten.
GPU1 only. Source quality checks and histories retained.
