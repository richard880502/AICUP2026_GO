# Result directory

Only artifacts required by the current dense-96 production pipeline and its
holdout reproduction remain at this level:

- `d96*_seed*.pt`: dense KataGo/HumanSL Transformer ensembles.
- `move*.pt`: move-sequence models.
- `resnet5*` and `resnet8c*`: production ResNets and inference snapshots.
- `val_probs_{d96,move,resnet5,resnet8c}*.npz`: validation predictions used
  to fit fusion weights and rank offsets.
- `snap_resnet*.npz`: saved per-epoch probabilities used by evaluations.
- `aggregator*.pkl`: optional historical question aggregators; disabled by
  default because they did not improve the independent holdout.

Large superseded checkpoints and ablation outputs are preserved under
`../archive/`; they are not needed for `predict_test.py`.

New work belongs under `../experiments/<experiment_name>/results/`, not in
this directory.

2026-10-20: `val_probs_d96{,_h,_h2}_opp.npz` (the d96 models applied to the validation cohort's opponent decisions; members of the
new default), and the fresh-cohort retrains `d96_h2_seed*.pt`, `resnet8c_h2*` (exclude features/holdout2/exclude_h2.txt; only for
experiments/fresh_cohort). `val_probs_resnet8c_h2.npz` is the epoch 5-8 snapshot mean like `val_probs_resnet8c_h.npz`.
