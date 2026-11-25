# A/B screening: small uncertain utility change, no pair-AUC improvement

Completed 2026-11-08. The preregistered mining gate failed; stop this
configuration without C, full C1+C2 refitting, or outer confirmation.

## What was actually trained

Two FROM-SCRATCH single-seed own-player Transformers, same production
architecture and ordinal head. A uses NLL; B adds player-level 1D/2D
pairwise loss with weight0.05 after five pure-NLL epochs. Both finish the
locked 15 epochs x64 steps, batch128, and select epoch12 by inner NLL.
Their first five epochs' losses and inner metrics match exactly.
Re-running both complete training arms from the final source reproduced
all history metrics and every tensor of both selected checkpoints exactly
(wall-clock timings excluded).

The merged cache has 51,211 unique games and 7,607 players, no observed
inconsistent player labels. The dense cohort has 2,000 players with
20 games each after merging reused games. Train has 7,007 players,
including 1,400 dense players; inner and evaluation each have 300 players,
30 per rank. Every player's games stay in one partition. No pretrained
encoder, official test data, or Public score enters fitting.

## Results

| Set | A utility | B utility | A pair AUC | B pair AUC |
| --- | ---: | ---: | ---: | ---: |
| Train diagnostic, 300 dense players | 0.615045 | 0.612938 | 0.663333 | 0.664444 |
| Inner, 300 players | 0.613283 | 0.620485 | 0.750000 | 0.753333 |
| Evaluation, 300 players | 0.597307 | 0.599414 | 0.643333 | 0.643333 |

Pair AUC is conditional 1D/2D odds on the true pair only; utility covers
all ten ranks. Inner is used to select epochs and is not independent.
Evaluation players are unseen by both new models, but this source cohort
has previously been studied, so this is exploratory evidence.

Evaluation utility delta B-A is +0.002107, paired 5,000-player-bootstrap
95% interval [-0.009309, +0.012997]. Exact accuracy changes 43.67% to
44.00%; >=2-rank error remains12.67%. Ten decisions change: five utility
gains, four losses and one equal-utility change. Pair AUC does not improve,
so the preregistered condition for mining is false despite positive utility
point estimates. No claim of mining efficacy can be made: C was not run.

These single-model scores are NOT comparable to production C1+C2 utility
0.717012 on the different 2,000-player cross-fit cohort. There is no
measured production improvement in this experiment. The fixed small
training budget and one seed also limit conclusions about other schedules.

## Mechanism and implementation verification

Three unittest cases cover ordinal sampling, disjoint training players,
5–20 unique games per bag, monotonic pair loss, invalid shapes, nonzero
encoder gradients and exact lambda0 identity. Each auxiliary step uses
four distinct players per pair rank. Data loading and evaluation validate
the saved player split. Both arms share initialization and NLL sample RNG;
the pair branch adds stochastic forwards after warmup, so later dropout
realizations need not match.

Selected-model gradient diagnostics on a fixed training pair batch show
the weighted pair gradient is about1.05% (A) /1.13% (B) of NLL gradient
norm on the input projection weights. This is a single-batch diagnostic,
not a global gradient ratio. It confirms a real but modest auxiliary
signal; it does not justify retuning on the observed evaluation outcomes.
Separate data loaders never detach the pairwise training forward.

The full report has per-rank and game-count utility, training loss/gradient
norm history, selected epochs, per-player predictions, confidence intervals
and all gate conditions. The standalone `training_curves.png` shows inner
utility and AUC over the fixed training budget.

## Completion and conditional work

- Data inventory, complete dense-game reconciliation and player exclusions:
  `data_audit.json` and `splits.npz`.
- Fixed architecture, budget, loss, sampling and stop rule:
  `PREREGISTRATION.md`, written before A/B.
- Differentiable losses, sampler, tests and complete A/B training: source
  files, A/B checkpoints and 15-entry histories.
- All-rank player-level evaluation, bootstrap and failure decision:
  `report.json`, with `gate.run_C=false`.
- Mining, full fusion and new outer confirmation: conditional prerequisites
  failed; not scheduled under this preregistration.

The previous frozen-embedding specialist remains relevant preliminary
evidence (pair AUC0.701925 versus baseline0.725725); this experiment tests
encoder gradients directly rather than repeating that frozen classifier.
