# Direct mining experiment after mixed A/B evidence

Locked 2026-11-08 before C training. The user explicitly requested trying
mining despite the preceding gate failure. This is a new exploratory
experiment, not a retroactive pass of the previous gate. Preserve both
earlier reports and all A/B checkpoints.

Reuse seeds 20261118/20261119/20261120, original player splits and fixed
questions, from-scratch architecture, normalization, 60 epochs x64 updates,
batch128, optimizer/schedule, pair weight0.05 and tau1. Train C only; use
the existing matched A/B controls. Select C by minimum inner question NLL,
as for A/B. Do not change epoch selection based on the earlier findings.

For the first five epochs use NLL only, sampling but ignoring random bags
exactly as A/B did; require exact warmup history parity. From epoch6,
sample four distinct 1D players and four distinct 2D players per step.
For each pair independently request semi-hard with probability0.30, else
random. Refresh training-only mean player scalar strengths over all dense
games at epochs6,8,...,60. Mining uses frozen scores to select a pair with
0 < strength_high-strength_low <=1, using sorted high-strength values and
searchsorted; never enumerate the full Cartesian product. Exclude players
already used within this auxiliary step. If no eligible pair exists, use
random adjacent players and record fallback. Draw5–20 games per selected
player without replacement and recompute scalar means WITH gradients.
No validation/evaluation labels or predictions enter mining.

Primary: mean paired C-B utility change over the three seeds on the same
300 evaluation players, with5,000 player-bootstrap resamples AFTER averaging
each player's seed changes. Also report C-A, all per-seed results, inner
results, exact/large errors, pair AUC, per-rank/count utility, changed
decision wins/losses and fixed probability-ensemble secondary results.
Apply Bonferroni97.5% intervals to C-B and C-A primary-style comparisons
as an additional two-comparison safeguard; ordinary95% intervals remain
descriptive. Earlier repeated hypotheses still limit confirmation.

Evidence supporting further work requires evaluation C-B mean utility>0,
at least2/3 seed utility gains, interval lower bound>0 for BOTH C-B and
C-A under the97.5% intervals, pair AUC not below B, large-error increase
<=0.01, and worst mean per-rank loss>=-0.05. Inner deterioration is reported
explicitly. Failure ends this recipe; no weight/mining-fraction/margin
search on this cohort. Production adoption always requires a later locked
full-fusion study and untouched outer confirmation.

Before training test training-only eligibility, positive-margin window,
within-step unique players, random fallback and differentiable recomputation.
Log cumulative random/mined/fallback counts and refreshes. Verify all
three C budgets/checkpoint selections, warmup identity against A/B and
the identical fixed query arrays. Output:
production/semi_hard_mining_direct_20261108/.
