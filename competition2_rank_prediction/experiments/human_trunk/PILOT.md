# Frozen HumanSL trunk pilot — preregistered 2026-11-09

Feasibility gates passed before any rank outcome: strict raw/SWA loading,
384×19×19trunk output, raw-export policy matching on8games/64positions
(both colors), and fixed-sample throughput. Raw weights selected because
they match the installed export; SWA is NOT selected by rank accuracy.

Corrected benchmark: ~1,009positions/s GPU1,~247positions/s CPU feature
construction,310,916,096bytes PyTorch peak allocation on GPU1. The initial
benchmark's zero-memory field read default GPU0 and is invalid; corrected
output is production/human_trunk_benchmark_v2_20261109/benchmark.json.
These figures exclude engine subprocess memory and full CUDA-context overhead.

## Frozen representation and data

- Original sparse train15,211games plus original validation8,000games.
  No official tests. Original model/CSV untouched.
- Sixteen evenly spaced non-pass own decisions, exactly the existing
  sampler. Fixed neutral1d metadata for EVERY example. No rank labels,
  player IDs or game IDs are passed into the neural input worker.
- Per decision:384global spatial means +384features at the actual played
  point +1pass indicator,769dimensions. The non-pass sampler makes the
  pass indicator zero in this pilot; it remains defined for capability tests.
- Original raw pretrained network frozen, no fine-tuning or new architecture.
- Eight CPU feature workers, bounded16jobs in flight; GPU1 batches up to
  four games/64positions. Spawn workers; never fork a CUDA-initialized job.
- Store fp16embeddings and explicit masks. Invalid SGFs produce an error
  record and masked game, not an invented valid zero vector.
- Keep source/model hashes, input metadata, complete shard coverage and
  throughput. Do not edit the extractor mid-run or mix incompatible shards.

## Fixed first classifier, before any rank scores

1. Standardize embedding channels using valid TRAIN positions only.
2. Fit randomized PCA64 on up to32,768TRAIN positions, seed20261109;
   never fit PCA/scaling on validation players or official tests.
3. Per game: phase scales1/2/4, mean+std per phase of PCA scores;
   append coverage and pass fractions. Mask empty phases explicitly.
4. Multinomial L2logistic regression C0.1, max_iter2000,
   class_weight='balanced', seed20261109. No hyperparameter/feature search.
   Standardize aggregated game features using training games only before
   logistic fitting. Exclude wholly invalid training games. Wholly invalid
   validation games receive uniform probabilities, a neutral contribution
   to log fusion, with counts explicitly reported (not invented embeddings).
5. Evaluate as a fourth member added to original frozen C1+C2, retaining
   original own/opponent features for C1. Five rank-stratified player folds
   fit weights/C1 only on their fitting players. Original baseline must replay.
6. Primary: official utility delta versus original, player-bootstrap97.5%
   interval, repeats averaged within player; per-rank and query-size reports.
   Record candidate fusion weights and probability calibration as diagnostics.
7. Advance only for gain>=0.005, positive lower bound, no rank drop>0.05.
   Positive results remain exploratory because validation was studied before;
   fresh-player confirmation is required before production replacement.
8. Do not force a contribution or generate a new submission if the branch
   receives zero useful weight or fails the gate. No Public-driven tuning.

All artifacts under production/human_trunk_20261109; vendored source and
checkpoint are under ignored runtime paths documented in FEASIBILITY.md.
Goal "find a breakthrough" is not fulfilled by successful extraction alone.
