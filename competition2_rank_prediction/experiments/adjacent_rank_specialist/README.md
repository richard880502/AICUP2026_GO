# 1D/2D adjacent-rank specialist

This exploratory issue #1 pilot uses frozen own/opponent Transformer
embeddings to train a player-level linear probe. Read the fixed
[preregistration](PREREGISTRATION.md) and [results](RESULTS.md).

From the repository root, with the existing venv and research caches:

```bash
export PYTHONPATH="$PWD/competition2_rank_prediction"
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.adjacent_rank_specialist.extract \
  --device cuda:1
runtime/competition1_research/.venv/bin/python -m \
  competition2_rank_prediction.experiments.adjacent_rank_specialist.run
```

Use `--output` with a new path for another run. Extraction rejects an
existing directory; the evaluation refuses to overwrite its report. Local
ignored output `production/adjacent_rank_specialist_20261107/` holds pooled
embeddings, checkpoint SHA-256 hashes, per-player predictions and the full
JSON report. Input data, checkpoints and binary results are not in Git.

The import path is required by the existing Transformer module's legacy
imports. The 2,000 players were already studied and are not a new outer
validation set. Exclusion is checked against the same manifests used by
the original cross-fit training scripts; those manifests document the
intended checkpoint training exclusions, rather than proving checkpoint
provenance independently.
