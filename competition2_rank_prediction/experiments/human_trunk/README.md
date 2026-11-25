# Frozen HumanSL trunk investigation

Completed outcome:[RESULTS.md](RESULTS.md). Both the fixed PCA/linear
probe and separate original-architecture temporal probe fail the utility
advancement gate. No new submission or production replacement.

Feasibility:[FEASIBILITY.md](FEASIBILITY.md). Protocols:
[initial gates](PLAN.md),[linear pilot](PILOT.md),
[separately registered temporal follow-up](TEMPORAL_PLAN.md).

## Resource layout

- Checkpoint: runtime/competition2_research/models/human_trunk.
- Pinned unmodified vendor source:
  runtime/competition2_research/vendor/KataGo-human-v1.15.0.
- Main features/linear probe: production/human_trunk_20261109.
- Temporal probe: production/human_trunk_temporal_20261109.
- CPU smoke: production/human_trunk_cpu_20261109.
- Corrected memory/throughput: production/human_trunk_benchmark_v2_20261109.

Only raw pretrained weights match the installed engine export. Actual
network metadata is fixed1d for all examples, never actual rank. Workers
receive SGF/color/config only; player IDs and labels are attached to saved
features for supervised downstream fitting and disjoint-split auditing.

## Original executed commands

From repository root, with `PYTHONPATH=competition2_rank_prediction` and
`OMP_NUM_THREADS=4`, using runtime/competition1_research/.venv/bin/python:

```bash
python -m competition2_rank_prediction.experiments.human_trunk.smoke \
  --device cpu --output production/human_trunk_cpu_20261109
python -m competition2_rank_prediction.experiments.human_trunk.equivalence
python -m competition2_rank_prediction.experiments.human_trunk.benchmark \
  --output production/human_trunk_benchmark_v2_20261109
python -m competition2_rank_prediction.experiments.human_trunk.extract \
  --workers 8
python -m competition2_rank_prediction.experiments.human_trunk.classifier
python -m competition2_rank_prediction.experiments.human_trunk.evaluate
python -m competition2_rank_prediction.experiments.human_trunk.temporal
python -m competition2_rank_prediction.experiments.human_trunk.evaluate \
  --output production/human_trunk_temporal_20261109
```

These are the original commands, not instructions to overwrite saved
results. Completed destinations are intentionally refused. Extraction
resumes only compatible source/model-hashed shards. Read saved reports
for quick reproduction of the completed comparison; full reruns need
separate output paths and the documented frozen feature inputs.

The baseline cache path contains `rank_joint_20261108`, but only original
validation d96/ResNet/opponent/final-position fields are read. Neither
Joint nor official-test predictions are evaluated here. The400-player
validation has been researched historically; no fresh-data claim is made.

## Checks

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/human_trunk
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/human_trunk
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.human_trunk.test_bridge
```

No pylint suppressions. Scoped type checks exclude unmodified imported
legacy/vendor code and absent dependency stubs. Tests include fixed
metadata, vendor namespace restoration, policy legality/normalization,
masked phase moments, no-decision masking, and bitwise equivalence of
temporal updates/checkpoint selection to original training helpers.
