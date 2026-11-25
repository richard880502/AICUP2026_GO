# Frozen HumanSL trunk probes — completed 2026-11-09

## Decision

Both fixed probes fail the advancement gate. **No confirmed breakthrough,
no production replacement, no new submission.** Original Public-best
C1+C2 remains0.7227214. The following scores are internal calibration CV.

| Full original C1+C2 plus candidate | Utility | Delta vs original | Player97.5% interval |
| --- | ---: | ---: | --- |
| Original baseline | 0.710606 | — | — |
| Frozen trunk PCA64 +linear probe | 0.711679 | +0.001073 | [-0.000856,+0.002999] |
| Frozen full trunk +3-seed original temporal encoder | 0.710797 | +0.000191 | [-0.001560,+0.001817] |

Neither reaches the fixed+0.005threshold or has an interval above zero.
The temporal follow-up was separately registered AFTER the linear result;
do not portray both as independent preplanned confirmations or select only
the better one. See[PILOT.md](PILOT.md),[TEMPORAL_PLAN.md](TEMPORAL_PLAN.md).

## What was actually extracted

- Official raw checkpoint, pinned upstream model/board/metadata code.
  Strict weight loading, not a guessed architecture or reimplemented trunk.
- Real `trunkfinal`,384×19×19; descriptor=384global means+384played-point
  activations+1pass flag. Fixed neutral1d metadata for all ranks; SGF and
  color only enter feature workers. The non-pass sampler leaves pass flag0.
- Train15,211games,7,607players; validation8,000games,400different players.
 16non-pass decision positions/game where available,100%usable games,
  no invalid inputs. Eight spawned CPU workers fed GPU1; bounded16jobs.
- No actual rank labels/player IDs enter upstream neural metadata.
  Labels are used only for the downstream classifier and calibration.

## Feasibility and limitations

On one fixed game/eight positions, raw weights match the installed engine
with mean total variation0.000441; SWA0.016983fails tolerance. Raw selection
was based on export equivalence, not rank accuracy.

Extended check:8games/64positions, both colors; corrected run mean TV
0.000673,max0.010298, below fixed0.005mean/0.02max tolerances. This is a
small numerical compatibility check, not bitwise equivalence for every SGF,
nor proof that all discrepancies are solely FP16. Rare-position difference
near1%TV is retained in the report, not hidden.

Corrected throughput ~1,009positions/s on GPU1 and~247positions/s for
single-process CPU feature construction; PyTorch peak310,916,096bytes
(~297MiB), excluding CUDA context/engine subprocesses. Initial benchmark
read default-device memory and returned0bytes; that field is INVALID.
Use production/human_trunk_benchmark_v2_20261109/benchmark.json instead.
GPU0's vLLM process was never stopped or reconfigured.

## Probe details

**Linear:** train-only channel scaling and PCA64 on32,768training positions,
73.2132%explained variance; phase1/2/4means/std plus coverage/pass fraction
=898game features. Train-only game scaling; balanced multinomial logistic
C0.1. Original training/validation players disjoint. No hyperparameter search.
Conditional fusion weights for the branch:0.0505–0.0690, not normalized
percentages. Nonzero weight is not evidence of official-score improvement.

**Temporal:** no PCA;16×769tokens into original ordinal d128/4head/3layer
Transformer. Seeds0/1/2,40epochs, original NLL/AdamW/OneCycle/augmentation.
Channel normalization uses optimizer rows only; per-seed10%inner-player
split chooses minimum game NLL, earliest tie. Selected epochs8/8/9;
inner minima1.882014/1.872671/1.918539. All40metrics/seed retained. Full
trunk stays frozen. A CPU test proves the history-recording wrapper returns
bitwise-identical model states and selection info to the original trainer.
Temporal branch fusion weights0.0939–0.1229, but utility still fails to improve.

Both candidates are added as fourth members; original own/opponent
distributions remain the C1feature source. Five player/rank-stratified
calibration folds fit fusion and C1 only on fitting players. Original
baseline replays exactly0.7106057000201695. Repeated query differences
are averaged within400players before5,000bootstrap resamples.
The cohort has been studied before; results are exploratory, not Public
or fresh external confirmation. No claim that all hidden representations
are useless, only that these two recipes did not establish a useful gain.

## Confirmation-data audit

Read-only index data/games_with_color.sqlite contains1,000,000game rows
and8,007player IDs. Their IDs are fully covered by original train7,607
plus validation400. Adding known holdout/cohort exclusion files leaves0
additional never-used indexed players with>=20games. Do not casually
promise a new unused400-player cohort from this index. A new player-disjoint
retraining split can test robustness, but is not genuinely untouched data;
every upstream member must exclude its evaluation players.

## Artifacts and checks

- Linear/source features: production/human_trunk_20261109.
- Temporal: production/human_trunk_temporal_20261109.
- CPU loading/shape: production/human_trunk_cpu_20261109.
- Both contain model/probability/report artifacts; evaluation.npz and
  comparison.png preserve complete conditional comparisons.
- Source/model hashes and metadata in extraction_manifest.json. The vendor
  and checkpoint remain ignored under runtime, original `.bin.gz` untouched.
- pylint on experiment directory10.00/10; scoped mypy
  `--follow-imports=silent --ignore-missing-imports`10source files pass.
  Imported legacy/vendor code and absent dependency stubs outside scope.
-7unit tests pass; no pylint suppressions. Independent artifact audit
  checks hashes, coverage, finite storage, player isolation and scores.
- Original CSV SHA unchanged:
  `7f222f69ed81e7b94d14e2e6cb7a77528bba1275b9aa59278ec063b391b3e2f0`.

The broad breakthrough goal remains open. A separately identified gap is
that previous normal-value extraction hardcodes maxVisits1; even its
`depth3` is pipeline queue depth, not three-ply search. True search-derived
decision regret is distinct from both this frozen trunk and one-visit
policy/value summaries and warrants a separate feasibility pilot.
