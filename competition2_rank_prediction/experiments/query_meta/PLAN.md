# Player-safe nonlinear query calibration

## Hypothesis

The current C1 residual changes question logits only along one shared rank
axis using a linear function of query summaries. A small residual MLP may
capture rank-specific nonlinear interactions among the pooled C1+C2 logits,
opponent evidence, within-query disagreement and final-result summaries.

## Fixed protocol

- Use the current C1+C2 members: `d96`, `resnet8c`, `opp_scratch`.
- Generate meta-training inputs by five-fold player-disjoint OOF calibration
  on the 400 validation players, separately under both holdout3 upstream model
  folds. Each meta-training query's label is excluded from its base fusion/C1
  fit.
- Use the established20queries/player, with5-20games per query.
- Input is the fitted C1 question logits concatenated with existing C1 query
  features. The network is a residual MLP, hidden width32, GELU, dropout0.1,
  zero-initialized10-class output.
- Standardize from OOF meta-training inputs. Use AdamW, learning rate0.003,
  weight decay0.05, full-batch training, at most100epochs. Select the earliest
  best NLL epoch on the held-player fold4 of OOF meta-training; refit at that
  fixed epoch using all OOF rows.
- Fit the usual C1+C2 pipeline on all400 validation players for each outer
  holdout3 fold, then apply the fixed meta-MLP. Holdout3 labels are used only
  for final scoring.
- Advancement requires utility gain at least+0.003, paired-player95% lower
  interval above0, and positive deltas in both outer folds. No architecture,
  epoch or loss sweep.

The historical2,000-player holdout3 cohort has been used for earlier studies;
this is a fixed method-specific confirmation, not a pristine benchmark or
Public estimate.
