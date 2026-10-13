# Static-check baseline

Checked on 2026-10-13 (Asia/Taipei) while preparing the initial research
snapshot. No existing Python implementation was edited during repository
packaging; files were copied mechanically into task-specific directories.

## Competition 1

Commands were run from `competition1_player_identification/` using its
project-local virtual environment:

```bash
python -m pylint --jobs=4 <18 tracked Python files>
python -m mypy <18 tracked Python files>
```

- Mypy: **success, no issues in 18 files**.
- Pylint: **9.96/10**. It exits non-zero because it reports structural
  duplicate-code and a small number of complexity warnings across related
  experiment entry points; no import or syntax failure remains when executed
  in the correct environment.

## Competition 2

Commands were run from `competition2_rank_prediction/`:

```bash
python -m pylint --jobs=4 <50 tracked Python files>
python -m mypy <50 tracked Python files>
```

- Pylint: **8.62/10**, non-zero. The legacy production scripts predate the
  current team style rules and include long lines, large experiment functions,
  missing docstrings, and local dependency import warnings.
- Mypy: **62 errors in 24 files**. Major groups are missing pandas/scipy/sklearn
  stubs, local Competition 1 and tutorial imports, untyped legacy containers,
  and several genuine assignment/indexing annotations that should be repaired
  before those files are modified further.

The complete outputs from this packaging run remain local at
`/tmp/competition1_pylint.log` and `/tmp/competition2_pylint.log`; transient
machine paths are not committed. The repository hook runs both tools on future
staged Python changes from the correct project directory and blocks the commit
when checks fail.

## Risk statement

The initial snapshot preserves the exact research code that produced the
documented results, including its static-analysis debt. It should not be
interpreted as a fully typed production package. The numerical reports and
locked evaluation protocol are preserved, while datasets, trained weights,
and licensed third-party assets are deliberately excluded.
