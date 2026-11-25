# Larger-budget three-seed replication: mixed utility signal

Completed 2026-11-08 under the separate
[multi-seed preregistration](MULTISEED_PREREGISTRATION.md), after the user
requested continued investigation of the initial pilot. All six models
finished 60 epochs x 64 updates, four times the initial budget. Pair weight
stayed 0.05; model, data and player split remained fixed.

## Primary results

Each row averages paired seed outcomes on the SAME 300 players. Inner
players select the epoch by minimum question NLL; evaluation players are
unseen by the trained models but this cohort has already been explored.

| Set | A NLL-only utility | B pair-loss utility | Delta | 95% player CI |
| --- | ---: | ---: | ---: | --- |
| Inner | 0.581470 | 0.577077 | -0.004393 | [-0.019079, +0.009489] |
| Evaluation | 0.579449 | 0.589334 | +0.009885 | [-0.003333, +0.023921] |

Evaluation seed changes are +0.011226, -0.010881 and +0.029309. Two of
three improve. The interval averages the three differences WITHIN player
before 5,000 player resamples; it does not pretend 900 predictions are 900
independent players. It is conditional on the three fitted seeds and does
not estimate all possible training-randomness uncertainty.

| Evaluation metric, seed mean | A | B |
| --- | ---: | ---: |
| Exact | 44.33% | 45.44% |
| >=2-rank error | 18.67% | 17.89% |
| NLL | 1.51354 | 1.51030 |
| 1D/2D conditional pair AUC | 0.672593 | 0.668519 |

The evaluation utility point estimate is encouraging, but its interval
crosses zero and the targeted pair discrimination declines. Inner utility
declines for all three seeds, and inner B-rank utility drops 0.06845 on
average, crossing the fixed safety threshold. Consequently
`report.json: gate.run_C=false`. The separately preregistered gate fails;
no semi-hard C, full C1+C2 fusion refitting, or new outer evaluation is
scheduled for this recipe. These results do not establish that pairwise
supervision has no possible benefit.

## Secondary ensemble and interpretation

The predeclared probability-average ensembles score 0.578568 (A) and
0.588568 (B) on evaluation, delta +0.010000 with interval
[-0.008428, +0.029655]. They also decline in pair AUC: 0.675556 to 0.666667.
Ensemble results do not replace the primary seed-mean analysis.

Training longer did not automatically yield higher decoded utility. On
inner only, maximum utility occurs at epochs 5–7 (about 0.626–0.636),
while minimum NLL selects epochs 47–51 (about 0.576–0.585). This descriptive
observation was not used to reselect checkpoints or obtain extra evaluation
scores. It suggests that epoch selection and downstream probability
correction deserve separate study before attributing any change to mining.
It does not prove that an early checkpoint would generalize better.

First-round 15-epoch scores use a different seed, so their absolute values
cannot isolate the effect of increased budget. Neither round's own-branch
scores can be directly compared with the full C1+C2 value 0.717012 on the
different 2,000-player cross-fit protocol.

## Verification and artifacts

- All three seeds use the first-round train/inner/evaluation player lists;
  the splits have zero pairwise player intersections.
- Each arm finishes 3,840 updates per seed. The first five warmup histories
  match exactly within each seed.
- Minimum-inner-NLL checkpoint selections are A/B: 47/51, 51/48, 48/49.
  Checkpoint seeds and recipe metadata match the fixed protocol.
- Seven tests cover differentiable ranking, ordinal player sampling,
  zero-weight identity, invalid budgets, player-level bootstrap input and
  keeping weak seeds in the average. Python lint/type checks pass as
  documented in README; dependency typing remains limited.
- Runtime `production/semi_hard_mining_multiseed_20261108/` contains
  `protocol.json`, `splits.npz`, six selected checkpoints, six complete
  histories, per-player seed predictions, `report.json` and
  `multiseed_training_curves.png`. The JSON includes per-rank and game-count
  changes and every gate condition. No official test/Public selection.

Potential follow-up would preregister utility-based inner epoch selection
or safe downstream calibration, with a new validation strategy. That is
a future hypothesis; it was not added after inspecting this evaluation.
