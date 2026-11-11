# Open-set v2 player matching

**In progress, 2026-11-11.** Phase-one controls have run; engine extraction and
open-set rich-profile training are underway. No final-holdout result or new
submission is claimed. Official Public remains **0.5236602**.

[Phase-one findings](PHASE1_RESULTS.md) · [Training and validation figure](figures/phase1_summary.svg)

## Verified protocol

The actual training CSVs contain 1,000,000 games and 8,007 players. Reserve
600 mixed-color eligible players for model selection and another 600 for the
final report. The other **6,807 players / 827,101 games** remain eligible for
training after excluding evaluation game content. The first engine pass caps
training at 16 games per player, making **120,561 games** including both
9,000-game evaluation cohorts. This cap is a first-pass resource choice;
it does not mean features exist for every remaining training game.

Each evaluation cohort has 600 candidates and ten references per candidate.
Each player supplies nested 1/2/3/5-game queries, giving 2,400 questions per
cohort. Reference and query contents are distinct. Both colors are allowed
without imposing an artificial 50:50 query ratio. Bootstrap players rather
than the correlated individual questions.

Source/cache ID, color, rank and row count alignment is checked across all
1,000,000 records. Original input files have local SHA-256 fingerprints.
The locked split verifies **zero player overlap and zero game-content
fingerprint overlap** across train, validation and final holdout. Fingerprints
use every move, including beyond the model's 384-token cache, and ignore HD
so metadata differences cannot conceal identical analyzed move sequences.
The input SHA and locked split indices are verified on repeat preparation.

The original pointer-only manifest is superseded. The SQLite index's old
relative paths are not used. Preparation now reads the current CSVs directly:

```bash
runtime/competition1_research/.venv/bin/python -m \
  competition1_player_identification.experiments.open_set_v2.prepare
```

Output stays inside `production/open_set_v2_20261111/`; licensed data, models,
arrays and intermediate artifacts are ignored by Git. Canonical split:
`clean_splits.npz`; evidence: `source_audit.json`, `split_audit.json`,
`engine_inputs/export.json`. Superseded pre-audit runs are in the nested
`archive/pre_hd_audit/`, not mixed with the current run.

## Engine representation and resources

Source v1.1 includes HD0 through HD9: **207,390 / 1,000,000 games have HD > 0**.
No setup-stone AB/AW records were found in the source audit. As a controlled
comparison with the current best method, this experiment uses its established
moves-only Japanese-rules / 6.5-komi analysis proxy, not a claimed reconstruction
of true game rules. Do not infer missing stones or actual komi from HD alone.
[Official dataset version notes](https://github.com/AILAB-NDHU/aicup2026-go-dataset/blob/main/README.md)
confirm that v1.1 adds HD while keeping moves unchanged.

Use normal KataGo b18c384nbt plus HumanSL rank_5k / rank_1d / rank_5d,
64 sampled decisions, one visit, ownership transitions, and the production
9,198-dimensional multi-scale rich-profile aggregation. Existing historical
HumanSL caches cover only the old 400-player cohort and cannot replace this
new extraction.

GPU0 and GPU1 now claim shared immutable shards with `--dynamic-shards`.
OS file locks prevent simultaneous writers, survive interrupted workers
without stale ownership, and allow the faster GPU to do more work. The flag
is opt-in; the original production extractor's default allocation is retained.
Other users' vLLM, llama-server and training services are not stopped.
Global progress is the saved shard count; per-worker ETA is not a combined ETA.

`followup.py --engine-pid <pid0> <pid1>` observes the exact live extractors,
then checks completeness, builds rich profiles with eight CPU workers, runs a
frozen Fisher baseline, fits 512D / 1024D encoders, and selects validation-only
fusion. Incomplete terminated extraction is an error; the watcher never
silently restarts an extractor.

## Experiment sequence and status

1. **Protocol / controls: done.** Opening statistics, symmetry-invariant
   opening-prefix matches and both-player move encoder controls have run.
   Query-count, query-color and clustered-uncertainty reports are saved.
   The rich-profile control waits for extraction.
2. **Open-set rich-profile encoders: queued after extraction.** ArcFace plus
   supervised contrastive and episodic few-reference/few-query loss. Every
   normalization and classifier identity is training-only. Tiny training
   libraries use fewer distinct references/queries; singleton identities
   contribute ArcFace only. Evaluation always uses ten references.
3. **Raw sequence / opening fusion: first pass done.** Three-layer 128D
   move Transformer, attention pooling and normalized 256D game embeddings;
   both absolute and target-relative color are encoded, with D8 augmentation.
   Opening-prefix IDF matching covers lengths 4/8/16/32/64.
4. **Game-to-game matching: fixed aggregators implemented.** Centroid,
   all-pair mean, per-query top-three and log-mean-exp reference aggregation
   are compared on exactly the same games. Learned pair scoring remains to
   be implemented and evaluated after the rich-profile encoder is frozen.
5. **Clean rank prior: first pass done.** Freshly train Competition 2's raw
   four-layer 192D ordinal move architecture; 300 inner training players pick
   checkpoint and temperature. No style-validation identities enter rank
   training. This is the raw rank branch, **not full C1+C2**. Rank remains a
   soft fusion feature, never a hard candidate filter.

After the engine and learned-matching ablations, lock a complete configuration
and compare it with the fixed baseline on the final 600-player cohort once.
That final evaluation, independent confirmation, complete findings and any
new submission are **not yet done**. Validation-selected gains are exploratory.

## Reproduce the early controls

Run from the repository root, after `prepare`:

```bash
export PYTHONPATH=.:competition1_player_identification
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.opening
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.prefix
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.train --kind opening
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.train --kind sequence
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.rank_prior
runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.fusion
MPLBACKEND=Agg runtime/competition1_research/.venv/bin/python -m competition1_player_identification.experiments.open_set_v2.report
```

Completed encoder fits and immutable data inputs are not overwritten. To
rerun an encoder deliberately, choose a separate `--output`. Phase-one fusion
scores are snapshotted so later engine fusion cannot relabel the early figure.

## Checks

Run pylint and mypy on `experiments/open_set_v2` and the updated production
extractor; the local pre-commit hook repeats both. Run `python -m unittest
competition1_player_identification.experiments.open_set_v2.test_invariants`.
Tests cover content/player separation, disjoint episodes, singleton handling,
D8/color/pass invariants, exact contest scoring, ordinal normalization,
autocast compatibility, exclusive shard writers and OS-lock recovery.

The Redmine infra document is unavailable in this session. The configured,
executable local hooks are used; verification against that original document
is not claimed.
