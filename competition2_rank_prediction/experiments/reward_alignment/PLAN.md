# Official-reward alignment, 2026-11-08

Reference: original C1+C2, reported Public 0.7227214. Joint's reported
0.7195608 is rejected. Never overwrite original checkpoints/submission.

Earlier loss/head tests already rejected reward-dominant training. This
experiment instead retains ordinal NLL and adds **0.2 × (1 − expected
official reward)**. Official reward is 1 for exact, exp(-1) for adjacent,
zero otherwise, NOT the Competition 1 Top-5 metric.

Fixed design before training:

- Original own-side d96 architecture/features, 40 epochs, batch 256,
  AdamW lr 0.0005, weight decay .05, OneCycle, noise .05, token dropout .15.
- Seeds 0/1/2, identical initialization, batches and augmentation across
  NLL and mixed-loss runs. No Joint/mining; frozen ResNet and opponent C2.
- Three arms: NLL/min-inner-NLL, NLL/max-inner-question-utility,
  mixed/max-inner-question-utility. Earliest checkpoint wins ties.
- Each seed's original player-disjoint 10% inner split; normalization
  fitted on optimizer training rows only. Supplement held-out inner players
  with their already extracted dense-library games (never optimizer rows).
  Fixed 20 synthetic questions per dense inner player, sizes cycling
  5/10/15/20. Sparse players contribute NLL but not question utility.
- Outer original 400 validation players excluded from all neural training
  and checkpoint selection. Outer data have been studied historically:
  this is exploratory evidence, not a fresh confirmation cohort.
- For each arm, replace d96 (do not add a fourth model). Refit identical
  geometric fusion and C1 with five-fold player-disjoint calibration.
  Also compare to original frozen C1+C2. Same question sampling throughout.
- Report utility, exact, NLL, per-rank/count scores, seed consistency,
  paired player bootstrap with repetitions averaged within player,
  97.5% intervals for two planned contrasts against NLL/min-NLL.
- Adoption requires positive adjusted lower bound, >=2/3 positive seeds,
  and no rank utility drop >.05. Otherwise retain original baseline.
  No Public-driven hyperparameter search and no automatic submission.

Artifacts live under production/reward_alignment_20261108, ignored by git.
Source, tests, findings stay here. GPU1 only; GPU0 vLLM is untouched.

Before any outer evaluation, the initial setup was stopped at seed0 NLL
epoch22: every original inner player had only two games. This incomplete
setup is archived as reward_alignment_sparse_setup_20261108 and is not a
result. Dense-inner supplementation above fixes the query-size mismatch;
no hyperparameters were selected from this interrupted run.
