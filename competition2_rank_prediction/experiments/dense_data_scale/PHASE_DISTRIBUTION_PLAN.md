# Fixed phase-distribution screen, 2026-11-09

Written before fitting or scoring these forests. This is a distinct probe of
the original d96 representation, not another coverage scaling experiment.
Sources are grouped here alongside the shared coverage scoring utilities.

Existing evidence: the old HGB used mean/std phase profiles; A/1D boundary
error-profile experiments and pooled-probability bag-distribution experiments
failed. This screen instead fits a full ten-class per-game classifier to
all original15,211 training games from7,607players, then asks whether adding
distribution shape helps a matched fixed nonsequence classifier. There is no
claim that distribution features are intrinsically new.

- Same four-profile96-decision cache, original400-player validation excluded
  from all tree fitting. No test or other-task candidate data.
- Channels:12raw plus3profile-centered surprisals. Fixed1/2/4phases.
- Control: masked mean/std,210dimensions.
- Rich: same means/stds plusq10/q25/median/q75/q90/skew,840dimensions.
- Both: ExtraTrees128trees, depth20, minleaf8, max_features0.5,6CPUworkers;
  seeds0/1/2. Inverse-player-game-count weights normalized to mean1.
- No parameter, phase, threshold or feature-family grid against validation.
- Primary: probability-ensemble whole-question utility, original fixed queries.
  Paired-player95% bootstrap, all seed deltas, per-rank and N breakdowns.
- Advance only for delta>=0.005, positive interval lower bound,2/3positive
  seeds and worst rank loss>=-0.05. Then retain all original C1+C2 members
  and test the fixed rich branch as an addition under player-disjoint
  calibration. Single-branch improvement never authorizes replacement.
- Historical validation is exploratory. No automatic submission or adoption.

Source: phase_features.py, phase_train.py, test_phase_features.py. Masked
padding, empty phases, moments and quantiles tests pass. Pylint10/10 and mypy
pass for all three modules. Runtime outputs use the isolated directory
`production/phase_distribution_20261109/`. GPU0 remains untouched.
