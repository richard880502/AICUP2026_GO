# Second-round multi-seed replication and conditional mining

Locked 2026-11-08 BEFORE fitting the second-round candidates. The user
requested continued research after the first-round stop. That stop remains
the first-round decision; this is a separately fixed larger-budget study.

## Fixed experiment

Reuse the exact original player splits, normalization and one fixed query
per player. The evaluation cohort has already been examined; all results
remain exploratory. No Public or official test selection. Keep single
own-player branch architecture and data. Initialize from scratch for each
of seeds20261118/20261119/20261120. Within each seed A/B share initialization,
NLL batch sampling and 60 epochs x64 steps (four times the first budget).
Keep batch128, AdamW0.0005, weight_decay0.05, cosine schedule, original
augmentation, first5 epochs pure NLL, and pair weight0.05/tau1. Do not
search weights or epochs; select minimum inner question NLL separately for
each arm. Evaluate selected checkpoints only after all A/B fits finish.

The primary estimand is the mean of the THREE paired seed utility changes
on the same evaluation players. Report per-seed results, and bootstrap
players after averaging their paired utility differences across seeds.
Do not treat three predictions per player as three independent people.
Keep all seeds, never select a winning seed. Also report seed-ensemble
probability averaging as a fixed SECONDARY analysis, not a substitute
primary. Report inner/evaluation utility, exact, >=2 errors, pair AUC,
per-rank/count utility and changed decisions; training histories and curves.

## Gate to conditional mining C

Advance to C only if mean B-A utility and mean 1D/2D AUC are strictly
positive on BOTH inner and evaluation, at least2/3 evaluation utility
deltas are strictly positive, mean >=2-error increase <=0.01, and worst
mean per-rank utility loss >=-0.05. Confidence intervals remain mandatory;
this gate permits exploration, not production integration. Failure stops
the second-round recipe, without post-hoc weight/budget search.

If eligible, train C for all three seeds with the SAME budget and warmup.
After warmup refresh current TRAINING-only player strengths every2 epochs,
using mean scalar strength over the available dense games. Four low/high
1D/2D player pairs per step: each pair is semi-hard with probability0.30,
otherwise random. Select only positive margins0 < s_high-s_low <=1, using
sorted strengths/searchsorted rather than the full Cartesian product.
Require unique players per auxiliary step and random fallback when no
eligible pair exists. Recompute selected bags with gradients; the miner
itself is detached. Log mined/random/fallback counts. A/B/C share data,
initialization, epoch criterion and budget. Compare C-B and C-A separately.

For any eventual full C1+C2 experiment require an exploratory mean utility
gain with positive95% lower CI versus A, consistent >=2/3 seeds and safe
large errors/per-rank changes; if C qualifies, it must also improve B
before attributing benefit to mining. Full fusion and new outer confirmation
would require a subsequent locked protocol and upstream-safe refitting.
This second round is complete when the fixed A/B replication and gate
are reported, plus C if its gate passes. It does not authorize adoption.

## Validation

Preserve existing sampler/gradient/zero-weight tests; add coverage for
configurable budgets/seeds and, if C executes, training-only mining,
positive-margin window, uniqueness and fallback. Check actual saved split
identity and completed budgets, selected epochs, warmup equality within
each seed, and required Python lint/type checks. Preserve all first-round
artifacts. Runtime output: production/semi_hard_mining_multiseed_20261108/.
