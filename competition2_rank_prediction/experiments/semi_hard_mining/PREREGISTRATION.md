# Single-branch ordinal pair-loss screening experiment

Locked 2026-11-08 before training A/B. This is a single-seed own-player
branch screening experiment. It is not a full C1+C2 replacement comparison.

## Dataset and exclusions

Merge existing feat96/train and dense_player_bags/features by game key.
Audit player-label consistency. Exclude any inconsistent player completely.
Eligible dense players must have at least 20 unique games after merging.
Stratify eligible players by rank, seeded permutation seed 20261108: first
15% inner validation, next 15% exploratory evaluation, remainder training.
Also keep sparse players in NLL training except excluded players. All games
of a player share the same split. Save player lists and data audit. Initialize
encoders FROM SCRATCH; no checkpoint has seen evaluation or validation.
The dense cohort has previously been studied and is not an untouched outer.

One fixed query per inner/evaluation player, 5–20 games without replacement,
seed 20261109, identical in all arms. Player-balanced NLL sampling includes
all ten classes. The pair branch uses only training dense 1D/2D players.

## Arms and fixed budget

Same RankTransformer as production: width128, heads4, FFN384, layers3,
dropout0.1, cumulative-link thresholds. Inputs are existing 96-decision
KataGo/HumanSL tokens. Training-only mean/std; no new extraction.
Seed20261108, AdamW lr0.0005, weight_decay0.05, gradient clip1,
15 epochs x64 steps. NLL batch128, token dropout0.15, noise0.05.
Cosine LR schedule, same starting weights/order/query RNGs across arms.

A: NLL only. B: NLL plus 0.05*softplus(-(s_high-s_low)), tau1,
after five pure-NLL warmup epochs. Four distinct 1D players and four
distinct 2D players per step; each draws an independent integer 5–20-game
bag without replacement; strength is mean per-game scalar score. Pair
forward retains gradients through the encoder. Each player occurs at most
once per auxiliary step. Epoch selection uses minimum inner question NLL
after unweighted geometric probability pooling. Record training losses,
gradient norms, inner utility/AUC and selected epoch. A does not need the
auxiliary forward; report timing as the unavoidable cost difference.

Mandatory tests: different players, correct adjacent labels, split exclusion,
loss monotonicity, nonzero encoder gradients, exact lambda=0 loss parity.

## Conditional C and stop

Evaluate selected A/B exactly once on exploratory evaluation. B must exceed
A in overall ten-rank utility and 1D/2D conditional pair-odds AUC on BOTH
inner and evaluation, with no >0.01 increase in >=2 errors and no per-rank
utility decrease worse than0.05. Otherwise stop without C or full fusion.
This generous screening rule only permits more study, never adoption.

If B passes, C uses identical architecture, budget and initialization:
70% random pairs /30% semi-hard after warmup, mining refresh every2 epochs.
Mining uses training mean strength over all available games, choosing
positive margins in (0,1]; cap each player's auxiliary use once per step,
random fallback when no eligible pair. Mining scores are detached; pair
forward is recomputed with gradients. Compare C against both A and B.
Only positive, consistent C/B evidence permits a separately fixed full
cross-fit C1+C2 evaluation with refitted postprocessing and then untouched
outer confirmation. Do not infer mining benefit solely from C>A.

Report all-rank utility/exact/adjacent/large errors, per-rank/count groups,
1D/2D AUC, train vs inner/evaluation pair metrics, per-player decisions,
5,000 paired player-bootstrap utility intervals and win/loss changes.
No official tests or Public leaderboard are consulted. Production remains
outside this experiment.
