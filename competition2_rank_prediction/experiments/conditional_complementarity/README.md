# Conditional Complementarity Audit (experimental)

Branch: `experiment/conditional-complementarity-v1` (based on
`experiment/ten-profile-production` so the existing P2 artifacts can be reused).

## Question

Ten HumanSL profiles helped an isolated d96 Transformer, but did **not** survive
the complete v47 pipeline (Issue #7). Is the **matched tenp10 minus tenp3**
residual complementary to **frozen original production v47**, particularly on
questions whose opponent identity evidence is less concentrated?

This branch **does not train a new encoder, does not alter any production model,
and does not claim a new holdout/leaderboard result**.

## Signals and safeguards

* Per-question identity-concentration proxy: mean top-two *class cosine gap*
  among HD0/HD1 opponent games from the exact saved `h3_sims_v47.pt` v4/v7
  similarities. The same holdout-class mask (`-5`) is applied as in
  `ten_profile_production.p2_evaluate.load_v47`. **Not calibrated known-player
  probability.** No labels or SGD are used to create this proxy.
* Primary population excludes any question containing HD>=2, because the
  **official test has zero HD>=2 games** (unlike holdout3). Run with
  `--include-hd2plus` for a clearly marked diagnostic.
* Compare **frozen original v47** with the additive, bounded log-probability
  residual `log(v47_tenp10) - log(v47_tenp3)`, *not* a direct replacement of
  production with the re-extracted/retrained ten-profile arm. Row-wise
  zero-center and clip the residual to [-2, 2]. This cancels matched-arm
  pipeline drift where possible, but cannot guarantee it removes all variance.
* Low/high concentration cutoff is the **median of the opposite fold's**
  unlabelled margin distribution. Train global (1 coefficient) and conditional
  (2 coefficients) fusion by **train-half NLL only**, on fixed nonnegative grid
  `[0, 0.25, 0.5, 1]`; evaluate held-out half once. Zero gives exact frozen
  baseline. There is **no post-hoc selection** between the two candidate arms.
* Player-grouped paired bootstrap CI, two fold results, changed/better/worse
  decisions, and boundary AUC for A/1D, 1D/2D, 2D/3D.
* Mandatory alignment of P2, original saved opponent question ids, and
  `question_folds.csv` (labels/folds checked). No positional assumptions.

**Critical limitation:** Existing v47 fold models were themselves trained
using the opposite holdout3 half. Thus although this probe fits its weights on
the *opposite* half, its training inputs can indirectly depend on the current
evaluation half's labels via that upstream fit. More importantly, holdout3 has
been repeatedly inspected. **This is exploratory / diagnostic, not an
independent outer evaluation.** The code always reports
`production_gate_passed: false`; even if its exploratory screen is positive,
a new nested, player-disjoint outer cohort (and official-coverage check) is
required before production adoption.

## Run (existing research host; no extraction or training)

From the `runtime/competition2_research` root after the P2 run in Issue #7:

```bash
PY=${PY:-c1/.venv/bin/python}
P2=/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production/p2
SIMS=/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt

$PY -m experiments.conditional_complementarity.extract_confidence \
  --sims "$SIMS" --output "$P2/identity_confidence.npz"

$PY -m experiments.conditional_complementarity.audit \
  --p2 "$P2/p2_report.npz" \
  --confidence "$P2/identity_confidence.npz" \
  --labels features/holdout3/question_folds.csv \
  --output "$P2/conditional_complementarity.json"
```

Outputs: JSON diagnostics and the paired `.npz` containing original/global/
conditional question probabilities and the exact eligibility/bucket masks.
For a distribution-mismatch diagnostic only, add `--include-hd2plus` and
write to a **different** output name. Keep the primary result untouched.

Run CPU synthetic regression tests from the repository root with:

```bash
python -m unittest competition2_rank_prediction.experiments.conditional_complementarity.test_audit -v
```

## Decision rule (exploratory screen only)

The conditional candidate must beat **unchanged production** by at least
`+0.005`, have a paired player-bootstrap 95% CI lower bound above zero,
and have no negative fold. This screen is *not* a production gate; otherwise
stop. Even a positive screen still needs a fresh, genuinely nested and
player-disjoint outer test. Do not tune confidence cutoffs or fusion grids
against holdout3 results.

Related: [Issue #7](https://github.com/richard880502/AICUP2026_GO/issues/7),
[Issue #6](https://github.com/richard880502/AICUP2026_GO/issues/6).