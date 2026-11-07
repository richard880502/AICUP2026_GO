# Frozen Region / Boundary Experts pilot

Independent experiment for GitHub issue #1. Uses the existing 2,000-player
C1+C2 OOF cache; no KataGo extraction or encoder training is required.

Completed pilot: [RESULTS.md](RESULTS.md). Follow-up train/OOF, calibration,
subgroup and feature-redundancy analysis: [DIAGNOSTICS.md](DIAGNOSTICS.md).
The diagnostics reproduce the pilot predictions exactly and support stopping
the frozen MLP expert expansion. Both reports are exploratory internal results.

The isolated terminal-winrate volatility follow-up is recorded in
[VOLATILITY_RESULTS.md](VOLATILITY_RESULTS.md): positive rank-error association
in both halves, but no utility improvement from the fixed scalar correction.

From the repository root:

```bash
bash competition2_rank_prediction/experiments/rank_experts/run_background.sh
```

The launcher runs in the foreground of its shell and can be managed by a
terminal multiplexer or a persistent job session. Output is automatically
redirected to `production/logs/rank_experts_20261106.log`; completed results
are `production/rank_experts_20261106/report.json` and `predictions.npz`.
It refuses to overwrite an existing result directory. A separate smoke run
used one epoch and ten bootstrap replicates; those results are discarded.

The fixed full experiment runs 200 epochs per fold and 5,000 paired player
bootstrap replicates. Models: bias-only, linear residual, fixed-gate region
MLPs and pair-mass-preserving boundary MLPs. Experts are evaluated using
five stratified folds within EACH upstream-excluded half, avoiding mixing
the two upstream model families across expert train and evaluation rows.

All scaler fits use expert training rows only. Every player has exactly one
question. Baseline probabilities, per-game own/ResNet/opponent predictions
and final-position features are frozen. See `PREREGISTRATION.md` for the
candidate definitions, stop rule, and exploratory-only status.

The cache's original exclusion provenance comes from
`runtime/competition2_research/experiments/crossfit_cohort/` and
`experiments/sgf_structure/base_probs.py`. The pilot checks each player is
in its half's exclusion manifest and game counts align; it does not refit
or independently reconstruct upstream training data. No official test
questions enter the experiment. Adoption requires a new outer cohort.
