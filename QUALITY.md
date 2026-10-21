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

## Competition 2 update (2026-10-21)

The opponent-side work (see `competition2_rank_prediction/README.md`) touched or added 22 Python files.
All of them were cleaned to the hook standard before committing, run together from
`competition2_rank_prediction/`:

```bash
python -m pylint --jobs=4 <22 staged Python files>   # 10.00/10, no messages
python -m mypy <22 staged Python files>              # Success: no issues found in 22 source files
```

- Pylint uses the default configuration (100-character lines). Inline disables are limited to
  single lines: `wrong-import-position` / `import-error` on project imports that must follow a
  `sys.path.insert` (pylint cannot resolve the project root from experiment directories), the
  checkpoint-compatible `RankTransformer` signature (`too-many-instance-attributes`,
  `too-many-arguments`), two long-lived handles in `PipelinedKataGo` (`consider-using-with`) and
  one torch false positive (`not-callable`).
- Untyped third-party imports follow the Competition 1 convention
  (`# type: ignore[import-untyped]`). The new `competition2_rank_prediction/mypy.ini` follows the
  untouched legacy modules (`train_eval`, `resnet_rank`, `move_transformer`, `rank_lib`,
  `blend_eval`, `board_sim`) silently so their documented debt does not block commits of the
  modules importing them; every staged file is still fully checked. `go_retrieval` (Competition 1,
  added to `sys.path` at run time) is marked `ignore_missing_imports`.
- The refactor was behaviour-preserving and verified by re-running:
  - `predict_test.py`: identical submissions on three configurations;
  - `transformer_rank.py`: identical training output, seeded CPU run against the previous version;
  - `fight_features.py`: identical features on two shards;
  - `make_holdout_test.py`: identical regenerated cohorts;
  - `extract_final_result.py`: identical SGF-derived values (KataGo search values vary run to run);
  - the experiment scripts: byte-identical result files, including the full pre-registered
    `crossfit_cohort/evaluate.py` report.
- Two exceptions:
  - `experiments/opponent_meta/run.py` was checked component by component against the old
    code (all bit-identical), not re-run end to end, since that takes about 8 hours on the
    shared machine.
  - In `experiments/strength_axis/run.py`, the ordinal-calibration CV predictions (`cv_C`)
    differ from the recorded run on one question. The unmodified original code reproduces
    the same new value, so the difference is an LBFGS numerical drift in the current
    environment, not the refactor.

## Risk statement

The initial snapshot preserves the exact research code that produced the
documented results, including its static-analysis debt. It should not be
interpreted as a fully typed production package. The numerical reports and
locked evaluation protocol are preserved, while datasets, trained weights,
and licensed third-party assets are deliberately excluded.
