# Pre-registration: 2,000-player cross-fitted confirmation (written 2026-10-21, before any holdout3 result exists)

## Why a new, larger cohort
The validation cohort, the old holdout and the fresh cohort (holdout2) have all been used for choosing methods, so none of them
can confirm the next candidates. Size: the per-question paired score difference has SD about 0.27 (fresh cohort,
shift residual vs default: 95% CI half-width 0.027 at n = 400). With n = 2,000 the SE is about 0.006, so the smallest effect
detectable with 80% power at two-sided alpha 0.05 is about +0.017 (about +0.018 with the Holm correction for two candidates).
A 400-player cohort would only detect about +0.038, so a +0.01-0.02 effect could not be confirmed there.

## Cohort and models
- `features/holdout3`: 2,000 players (200 per rank, >= 20 games, not in validation / holdout / holdout2), one 5-20 game
  question each (24,822 games), `scripts/make_holdout_test.py --players 2000 --seed 20261021`.
- Cross-fitting: two rank-balanced halves (`question_folds.csv`, `exclude_c{0,1}.txt`). Models `d96_c{j}`, `resnet8c_c{j}`
  (and any candidate model) are trained without half j and score only half j. Every holdout3 player is scored by models
  that never saw them. Holdout / holdout2 players are back in training (they are no longer test sets here).
- Fusion weights, offsets and every candidate's parameters are fitted on the validation cohort only, separately per half's
  models.

## Baseline
Current default: best2 + opponent member (`d96_c{j}`, `resnet8c_c{j}`, `d96_c{j}` on opponent moves), log-mean pooling,
validation offsets, expected-score decoding.

## Candidates (fixed now; nothing else will be tested on holdout3 for adoption)
- **C1 shift residual**: `experiments/opponent_meta/run.py` configuration `shift lam=0.01 [E1+E2+E3]` (the shift
  configuration with E1+E2+E3 that had the best validation CV), fitted on the validation cohort's 20 questions per player.
- **C2 opponent-specific Transformer**: a d96 model trained to predict the PLAYER's rank from the OPPONENT's decisions on
  the full training set. The variant (head-only fine-tune / full fine-tune / from scratch) is chosen on the validation
  cohort only and written below BEFORE holdout3 is scored. It replaces the borrowed opponent member in the default.
  - chosen variant (2026-10-21 10:25, validation only, `experiments/opponent_specific/select_on_val.py`): **from scratch**
    (d96 recipe, 40 epochs, 3 seeds, trained on `features/feat96opp/train` with the player's label). Validation question score
    0.6935 vs 0.6874 for the borrowed member (+0.006 [+0.001, +0.012]); full fine-tune +0.004, head-only -0.005.
    Cross-fitted models: `opp_scratch_c{j}` (exclude `exclude_c{j}.txt`).

## Decision rule
Primary metric: mean competition score over the 2,000 holdout3 questions; paired bootstrap over players (questions).
Holm correction over {C1, C2}: adopt a candidate only if its one-sided p-value passes Holm at alpha 0.05 (equivalently its
Holm-adjusted lower confidence bound is above 0). Effect sizes are reported regardless. The combination C1 + C2 is reported
as exploratory only.
