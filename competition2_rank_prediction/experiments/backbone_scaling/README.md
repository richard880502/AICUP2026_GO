# Dense Transformer scaling ablations

The completed single-seed variants were:

- `hp_e80`: 80 epochs; question log-mean/expected score 0.5595.
- `hp_big`: larger 192-d, 4-layer model; score 0.5771.
- `hp_reg`: stronger dropout and token dropout; score 0.5789.

The existing three-seed d96 ensemble scores 0.5906 on the same validation
protocol. Cross-fitted fusion assigns negligible weight to these new variants,
so none were retrained on the holdout-excluded cohort.

Checkpoints and reports are preserved under `results/`; logs are under
`logs/`.
