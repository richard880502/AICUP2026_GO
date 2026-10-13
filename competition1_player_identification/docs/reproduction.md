# Reproduction guide

Run these commands from the project root. The local virtual environment is
already provisioned at `.venv/`.

## 1. Verify the frozen best report

```bash
.venv/bin/python -m json.tool \
  results/best/arcface_cross_dimension_ensemble.json
```

## 2. Rebuild rich profiles (optional)

```bash
mkdir -p features/reproduced
.venv/bin/python cache_rich_profiles.py \
  --features features/best/humansl_multirank_s64 \
  --output features/reproduced/humansl_multirank_profiles.npz \
  --workers 12
```

## 3. Refit the selected metric ensemble

```bash
.venv/bin/python evaluate_arcface_cross_dimension_ensemble.py \
  --profiles features/best/humansl_multirank_s64/profiles.npz \
  --output results/best/reproduced_humansl_multirank_arcface.json \
  --blend-by-query-count \
  --device cuda:0
```

The model selection remains candidate-only; do not use query labels to choose
epochs, dimensions, blend weights, or auxiliary-score weights.

The frozen reference uses the same command through `./scripts/reproduce_best.sh`
and reports an independent proxy of 0.916153. The optional rebuild writes to a
new location so it cannot overwrite the frozen cache.

## Static checks

After Python modifications, run:

```bash
.venv/bin/pylint <changed_python_files>
.venv/bin/mypy <changed_python_files>
```
