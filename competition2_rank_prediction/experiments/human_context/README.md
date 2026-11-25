# Conditional HumanSL context pilot

This is a new feature experiment, not another architecture/loss sweep.
It tests whether human-profile value and short-term uncertainty add
information beyond frozen C1+C2 and normal-KataGo context.

See [fixed protocol](PLAN.md). Source definitions:
[KataGo analysis engine](https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md).
No claim is made that trunk embeddings are available through this API;
this experiment uses documented output heads, not hidden activations.
Completed outcome: [RESULTS.md](RESULTS.md); no significant full-pipeline
gain, not adopted.

## Run

From repository root, set `PYTHONPATH=competition2_rank_prediction` and
`OMP_NUM_THREADS=4`, then use:

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.human_context.probe
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.human_context.extract
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.human_context.evaluate
```

Probe defaults to `production/human_context_probe_20261109/`; extraction
and evaluation to `production/human_context_20261109/`. There are8shards,
1,000games each. A completed shard is atomically renamed and can be reused.
Settings include extractor source SHA; incompatible resume is rejected.
Evaluation refuses incomplete coverage or an existing final report.

Input paths are read-only: original validation SGFs and existing original
C1+C2 validation caches. No official test predictions/labels are used.
No new submission, commit or push is automatic. GPU1 only.

## Installed model provenance

KataGo v1.18.2, git `fd0723fdbc0e9d82cf269c9630af8c27c57c07c4`, CUDA12.1.105.
Existing files under `runtime/competition2_research/c1/models/katago/`:

- Normal: `kata1-b18c384nbt-s9996604416-d4316597426.bin.gz`, SHA256
  `9d7a6afed8ff5b74894727e156f04f0cd36060a24824892008fbb6e0cba51f1d`.
- Human: `b18c384nbt-humanv0.bin.gz`, SHA256
  `637746e44f0efe00ad1245a50aa9bbf0716efe364c43965ead97bd6835d84ab5`.

No model downloads or runtime upgrades are required. Normal b18 is used,
not the tf3 Transformer model. Black-perspective engine output is
explicitly converted to the labelled player's perspective. Uncertainty
keeps its sign when reversing colors; lead and winrate do not.

The baseline cache dependency is
`production/rank_joint_20261108/prepared.npz`, but **only original validation
d96/ResNet/opponent/final-position fields are read**. Joint predictions
and official-test arrays are not used in this experiment.

## Follow-up feasibility, not part of this output-head pilot

The [official v1.15.0 release](https://github.com/lightvector/KataGo/releases/tag/v1.15.0)
also publishes `b18c384nbt-humanv0.ckpt` expressly for PyTorch experiments
and fine-tuning. The release API was checked on2026-11-09: asset size
323,752,318bytes; URL:
https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.ckpt

This establishes a possible route to hidden representations without
reverse-engineering the engine binary. It does not verify loading,
feature equivalence, inference throughput or rank-prediction value. No
checkpoint has been downloaded as part of this pilot. Those checks need
a separate preregistered experiment if the current screen is negative.

## Quality checks

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/human_context
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/human_context
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.human_context.test_features
```

Mypy checks changed modules, not legacy imported code or unavailable
dependency stubs. No pylint suppressions. Tests cover perspective,
actual post-move alignment, absent response masks, invalid values,
normal/human feature layout and training-only missing-value imputation.
