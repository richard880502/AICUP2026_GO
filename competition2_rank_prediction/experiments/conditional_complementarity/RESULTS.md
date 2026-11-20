# Conditional complementarity audit — screen fails; the premise does not hold

Completed 2026-11-20. Exploratory only (`production_gate_passed: false` by design). No training, no extraction; reuses the
P2 question probabilities in `p2_report.npz` and the saved `h3_sims_v47.pt`. The three bundled unit tests pass.

## Decision

**Stop.** The matched ten-profile residual is not more useful where opponent-identity evidence is weak, and adding it to
frozen production v47 lowers utility in every fusion variant.

## The premise, tested directly

The question is whether `tenp10 - tenp3` helps more on questions whose identity evidence is less concentrated. Split at
the opposite fold's median identity margin:

| Stratum | Questions | `tenp10 - tenp3` | 95% CI |
| --- | ---: | ---: | --- |
| low identity margin | 787 | **+0.00194** | [-0.00288, +0.00676] |
| high identity margin | 800 | **+0.00171** | [-0.00257, +0.00612] |
| all eligible | 1,587 | +0.00183 | [-0.00136, +0.00508] |

The matched gain is the same size whether identity evidence is weak or strong. There is no conditional complementarity
to exploit, and this reading does not depend on the fusion design below.

## Screen against frozen production v47

1,587 official-compatible questions (413 containing an HD>=2 game excluded, matching the 413 found in `rank_context`).
Production on this subset: 0.76442.

| Candidate vs production | Delta | 95% CI | Halves | Changed (better / worse) |
| --- | ---: | --- | --- | --- |
| `tenp10` in place of production | -0.00103 | [-0.00501, +0.00279] | -0.0024 / +0.0003 | 28 (12 / 15) |
| global residual fusion | -0.00199 | [-0.00558, +0.00159] | -0.0049 / +0.0008 | 20 (7 / 12) |
| **conditional residual fusion** | **-0.00222** | [-0.00524, +0.00080] | -0.0049 / +0.0003 | 17 (5 / 11) |

Screen (`>= +0.005`, lower bound `> 0`, both halves `>= 0`): **fails on all three criteria**, with a negative point
estimate. Fitted weights on the fixed grid `[0, 0.25, 0.5, 1]`, by train-half NLL: fold 0 chose 1.0 for every weight;
fold 1 chose 0.5 global, 1.0 low-margin, 0.0 high-margin. Fold 0's choice cost 0.0049 on its held-out half.

Hard-boundary AUC barely moves (A/1D 0.8308 -> 0.8310, 1D/2D 0.7751 -> 0.7761, 2D/3D 0.8361 -> 0.8367).

## Why the residual hurts

`log(v47_tenp10) - log(v47_tenp3)` is taken at the v47 output, and each arm's v47 was **refitted separately**. P2
measured that refit alone moves v47 utility by about +-0.001 under a 1e-6 jitter of its input. The residual therefore
carries two independent refits' disagreement on top of whatever the extra profiles contribute. Shared extraction and
retraining drift cancels; refit drift does not. Train-half NLL then rewards fitting that noise.

A cleaner residual would be formed at the C1+C2 level and pushed through a single frozen v47 fit. It is not worth
building: P2 stage A already shows the C1+C2-level matched difference is -0.0005 [-0.0046, +0.0037], so there is
nothing for a single v47 fit to propagate.

## Where this leaves the ten-profile line

| Level | `tenp10 - tenp3` |
| --- | --- |
| d96 alone, validation (P1) | +0.0139 [+0.0072, +0.0208] |
| inside C1+C2, holdout3 (P2 A) | -0.0005 [-0.0046, +0.0037] |
| v47, holdout3 (P2 B) | +0.0022 [-0.0006, +0.0052] |
| v47, low identity margin (this audit) | +0.0019 [-0.0029, +0.0068] |

Closed. Do not extract ten profiles for the full corpus or the official test set.

## Reproduce

From `runtime/competition2_research` on the research host, after P2:

    P2=/mnt/share_data_77/richard/AICUP2026_GO/ten_profile_production/p2
    c1/.venv/bin/python -m experiments.conditional_complementarity.extract_confidence \
      --sims /tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt --output "$P2/identity_confidence.npz"
    c1/.venv/bin/python -m experiments.conditional_complementarity.audit \
      --p2 "$P2/p2_report.npz" --confidence "$P2/identity_confidence.npz" \
      --labels features/holdout3/question_folds.csv --output "$P2/conditional_complementarity.json"
