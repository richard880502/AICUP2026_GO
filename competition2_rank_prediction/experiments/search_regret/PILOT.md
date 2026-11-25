# Search-regret pilot — fixed before rank outcomes, 2026-11-09

API/cost probe succeeded on16fixed games/192pre/post positions: all32actual
visits, raw and searched root/child fields available, mean absolute searched
minus raw lead0.9833points. Warm throughput159positions/s gives a rough
13.4minute lower-complexity extrapolation for8,000games ×16positions.
Selected surprising middle-game moves may cost more; this is not a promise.

## Inputs and frozen selection

Original400validation players/8,000games only. Original C1+C2 members and
Public0.7227214submission untouched. Normal b18NBT, no HumanSL model,
Japanese6.5komi, deterministic symmetry and fixed seed, GPU1only.
Search-bot initialization uses `forDeterministicTesting=true`, with a
fixed NN seed. Asynchronous scheduling and floating-point kernels may
still prevent bitwise feature reproduction; cached source traces are kept.

For both labelled player and opponent, use cached normal-policy surprisal
and legality masks from their original96decision features. Prefer turn
indices20..min(250,game_length−10), highest surprisal first, lower turn
breaks ties. Choose up to4, filling remaining slots from other valid cached
decisions in the same order. No rank labels enter selection. Search every
selected position and its actual post-move position at maxVisits32.

## Fixed features

All value fields reported BLACK by engine; multiply differences by+1for
Black actor or−1for White actor, independently for own/opponent roles.

Per selected decision:

- Raw score loss=(rawLead_before−rawLead_after)×actor_sign/20.
- Raw winrate loss=(rawWinrate_before−rawWinrate_after)×actor_sign.
- Search score loss=(best_child_scoreLead_before−root_scoreLead_after)
  ×actor_sign/20; best child=lowest official order.
- Search winrate loss analogous using winrate.
- Cached surprisal; move index/(game_length−1).

These are estimated finite-budget regrets, not exact ground truth. Root
and child statistics differ and search noise may yield negative estimates.

Per actor role, raw7features: mean/max of raw score loss clipped0..2.5,
mean/max raw winrate loss clipped0..1, mean surprisal, mean move fraction,
number of valid selected moves/4. Search6features: analogous mean/max
searched score/winrate loss, mean signed(search−raw score loss) clipped
−2.5..2.5, and mean absolute clipped difference. No extra feature search.

Layout: own_raw7,opponent_raw7,own_search6,opponent_search6 =26features.
Missing decisions excluded; missing actor-role features stayNaN except
coverage0. Fold-training-only imputation; counts reported explicitly.

## Calibration and decision rule

Same original five rank-stratified player folds and20questions/player.
Fit fusion and C1 only on fitting players. Question features add means of
game descriptors; original own/opponent/final inputs retained.

Methods: original C1lambda0.01; regularization-only C1lambda0.1; raw14
controls+original C1lambda0.1; raw14+search12+original C1lambda0.1. Use
same300step shift model. No model/hyperparameter selection on outer data.

Two primary contrasts: full search vs original, and full search vs raw.
Average repeated query differences within player,5,000bootstrap resamples,
97.5%intervals for two contrasts. Advance only for >=0.005over original,
both lower bounds>0, and no rank loss>0.05. Raw-only gains are secondary,
not post-hoc proof of search value. Original baseline must replay exactly.

Historically studied cohort: exploratory, not Public or fresh confirmation.
The index has no wholly unused player IDs; robust confirmation requires
new predeclared player-disjoint retraining or new data, with honest reuse
limitations. No automatic production overwrite, submission, commit or push.
