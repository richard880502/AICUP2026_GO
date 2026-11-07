# First cached pilot: no gain

Completed 2026-11-06. Internal exploratory results, not official Public scores.
Source baseline hash, full metrics, player OOF predictions and fold assignments:
`production/rank_experts_20261106/{report.json,predictions.npz}`.

2,000 unique players, 200 per rank, one question each. Five stratified expert
folds separately within each upstream-excluded half; 200 fixed training steps
per fold, no parameter search. All upstream predictions and C1 fits frozen.
Zero-initialized corrections passed logit parity checks. The cached baseline
reproduces 0.7170117549. Runtime: 9.4 seconds on GPU 1, using cached features.

| Method | Utility | Exact accuracy | Error >= 2 ranks | Utility delta vs baseline [paired player 95% CI] |
| --- | ---: | ---: | ---: | --- |
| Frozen C1+C2 | 0.717012 | 0.5820 | 0.0510 | reference |
| Bias-only residual | 0.715248 | 0.5795 | 0.0515 | -0.001764 [-0.006953, +0.003293] |
| Linear bounded residual | 0.699789 | 0.5565 | 0.0540 | -0.017223 [-0.028469, -0.005925] |
| Fixed-gate Region MLPs (primary) | 0.695680 | 0.5500 | 0.0540 | -0.021331 [-0.033313, -0.009405] |
| Pair-mass-preserving Boundary MLPs | 0.707714 | 0.5670 | 0.0505 | -0.009298 [-0.015436, -0.003373] |

Bootstrap: 5,000 paired player resamples; secondary intervals are descriptive
and not multiplicity-adjusted. The primary region candidate also trails the
bias-only control by -0.019567 [-0.031687, -0.007923].

Conclusion: the preregistered pilot threshold is not met. Do not integrate or
submit these experts. The results are consistent with overfitting or
miscalibration when fitting flexible corrections on 800 players per fold;
they do not prove that every expert design must fail. The neural recipes use
cross-entropy, so lower classification loss would not itself imply higher
competition utility. No learned-gate or combined-model follow-up is justified
by this first result. Production remains C1+C2 (user-reported Public 0.7227214).

Scope limitation: the previously studied 2,000-player cohort is not untouched;
the exclusion audit checks player manifests and cache alignment, relying on
the existing upstream training provenance. No new outer cohort was trained or
evaluated. This is a low-cost screening experiment, not an adoption test.
