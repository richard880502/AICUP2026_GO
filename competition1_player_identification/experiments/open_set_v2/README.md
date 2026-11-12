# Open-set v2 player matching

**Completed, 2026-11-12.** All five directions, frozen inference replay,
configuration sealing and the independent 600-player final evaluation are done.

Final exponential Top-5: **0.295403** versus the fixed Fisher-profile baseline
**0.057627**. Paired difference **+0.237775**, player-bootstrap 95% interval
[+0.211479, +0.265015]. These are new-task proxy results; official Public
remains **0.5236602**, with no new competition upload.

[Complete results](RESULTS.md) · [Final figure](figures/completed_summary.svg) ·
[Phase-one controls](PHASE1_RESULTS.md)

## Replay or reproduce

```bash
bash competition1_player_identification/experiments/open_set_v2/reproduce.sh
```

With the existing sealed artifacts, this verifies model/data/code hashes and
recomputes saved final metrics without retraining or selecting another model.
On a fresh authorized-data workspace, it runs the full pipeline, including
32-epoch 512D/1024D fits, three engine-assisted rank seeds, three pair-feature
ablations, validation-only fusion, exact inference replay, sealing and final.
Retraining is subject to floating-point variation; serialized replay is exact.
The original `followup.py` was the eight-epoch first-pass watcher; use this
reproduction script for the completed experiment's full training budget.

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
frozen Fisher baseline and fits first-pass eight-epoch encoders. The full
completed pipeline is in `reproduce.sh`. Incomplete terminated extraction is an error; the watcher never
silently restarts an extractor.

## Completed experiment sequence

1. **Verified protocol and controls.** Opening statistics, exact D8 prefixes,
   raw-move Transformer and frozen rich-profile Fisher control. Player/content
   separation, color groups, query-count groups and bootstrap were checked.
2. **Open-set rich-profile encoders.** Both dimensions trained for 32 epochs
   of 256 episodes using training-only normalization, ArcFace, SupCon and
   few-query/few-reference matching. Validation selected epochs 13 / 20;
   final outcomes never selected checkpoints.
3. **Move sequences and opening fusion.** Raw-move control and exact opening
   matching were tested independently and as validation-selected fusion inputs.
4. **Distribution and learned game matching.** Four fixed aggregators plus
   frozen-encoder Top-20 hard-negative reranking. A/B game features each use
   twenty columns, both use thirty-three. Validation selected the 1024D
   matching family and a 0.75 reranker blend. Stage1 Recall@5/10/20/30/50,
   natural misses and all three ablations are saved.
5. **Clean C2 rank priors.** Raw-move ordinal control and a stronger,
   three-seed C2 decision-Transformer architecture retrained on the new 64x41
   cache. A 300-player inner training split selected/calibrated rank models.
   This tests C2's rank-model branches, not the full C1+C2 ensemble. Priors
   are soft; no candidates are discarded by a hard predicted-rank filter.

Serialized inference reproduced every selected validation source and baseline
with zero numeric error and zero rank discrepancies. The complete recipe and
artifact/code hashes were then sealed. Exactly that recipe scored 0.295403
on the other 600 players, versus 0.289407 on selection validation. The final
cohort is now reported and must not be reused as a fresh tuning target.

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
