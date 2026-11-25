# Game-coverage follow-up: design before extraction

## Evidence motivating a new test

The old d96 scaling curve stops at 15,211 games / 7,607 players, at most two
games per player. Source training contains one million games / 8,007 players.
An index audit found every anonymous player has exactly one source rank.
More source games cannot create more independent label-bearing players, but
can expose within-player variation absent from a two-game cache. Neither
that potential benefit nor full-data saturation has been established.

This differs from the raw-move identity experiment: that trained on many raw
games, not the production KataGo/HumanSL decision representation. It also
differs from question-supervision, which reused the limited cached games.

## Controlled experiment requirements

1. Preserve original validation IDs and keep every additional game from those
   players out of optimization, normalization and model selection.
2. Select additional games only from existing optimizer-player partitions.
   Keep the same player set across arms; do not conflate extra players with
   more games per player. Exclude game-key duplicates.
3. Start with fixed two-versus-eight-game-per-player coverage, capped by source
   availability. Use player-balanced sampling and identical optimizer update
   budgets, architecture and augmentations. This isolates game diversity from
   additional training compute.
4. Use the same feature definitions, engine models, decision count and handicap
   convention as production. Benchmark extraction first; estimate full cost
   before launching. Never use other-task candidate pools.
5. Reserve an inner player split for checkpoints; compare three matched seeds.
   Evaluate whole questions and the full frozen-other-branch C1+C2 pipeline,
   not only game accuracy. Report per-rank and per-query-size results.
6. Original 400-player CV is exploratory due to prior reuse. A positive result
   needs separately player-disjoint retraining confirmation before adoption;
   do not call another split of historically explored data genuinely fresh.

## Preflight results, 2026-11-09

The existing 7,607 optimizer-player IDs have no overlap with the 400 validation
IDs. Source counts allow 15,211 / 30,391 / 60,224 games at per-player caps
2 / 4 / 8. There are 7,348 players with at least eight source games; by rank
D through 6D: 847, 771, 958, 1351, 392, 569, 577, 600, 616, 667.
Thus an eight-game cap adds 45,013 games without adding player identities.

A real GPU1 extraction smoke test using the unchanged runtime extractor,
96 decisions and HumanSL profiles rank_5k/rank_1d/rank_5d completed 16 games
in 13.806 wall seconds (extractor-reported 1.20 games/s). GPU memory observed
was 2,436 MiB. GPU0's separate service was unchanged. Output shape matches
production: 16 x 96 x 4 x 3. Small-sample startup/batching effects mean this
is not a sustained-throughput guarantee. At 1.20 games/s, 45,013 additional
games take approximately 10.4 hours; older multiworker logs are slower per
worker. Benchmark representative larger batches before promising a deadline.

Smoke artifacts: `production/game_coverage_preflight_20261109/features/`.
These are re-extracted known training games solely for timing/schema checks,
not newly covered source games and not training results. Do not treat the
partial 16-game shard as a complete source shard or resume production into
that directory. No Python source was changed for this preflight.

## Nested pilot manifests

`coverage_manifest.py` generated isolated source shards in
`production/game_coverage_manifest_20261109/`: 50 training players per rank,
500 total; cap2 has 1,000 games and cap8 has 4,000. Selection is deterministic
by seeded SHA256 ordering, with at least eight source games required. An
independent check confirmed identical player sets, exact per-player counts,
no duplicate keys and strict cap2 inclusion within cap8. Validation IDs were
excluded by the generator. Source CSV hashes are recorded in `manifest.json`.
The recorded script hash refers to the generation-time revision; subsequent
lint cleanup extracted shard writing into a helper and changed the cached-ID
array conversion. It is not the current source hash. Reproduction should use
a new output directory and verify selected IDs/keys rather than rewrite this
historical record.

Checks: pylint 10/10; mypy passed for the new module. These manifests are data
selection artifacts only, not evidence of model improvement.

No expanded extraction or training has been launched. Public-best C1+C2 and
its submission remain unchanged. Next: create nested, duplicate-free source
game manifests for a fixed optimizer-player pilot and its disjoint inner split,
then run matched two-versus-eight-game training. Existing GPU0 services remain
out of scope.
