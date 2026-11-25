# Initial feasibility evidence — 2026-11-09

The preceding HumanSL output-head context pilot is complete and negative;
see[results](../human_context/RESULTS.md). This separate branch has now also
completed two fixed rank probes; neither establishes an improvement.
See[RESULTS.md](RESULTS.md) for the complete outcome and limitations.

## Verified

- Official release asset downloaded, exact323,752,318bytes:
  `runtime/competition2_research/models/human_trunk/b18c384nbt-humanv0.ckpt`.
- Recorded SHA256:
  `84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27`.
  This is a locally recorded content hash, not an independently published
  official checksum. Download source is the official GitHub release URL.
- Matching upstream release source cloned read-only to
  `runtime/competition2_research/vendor/KataGo-human-v1.15.0`, pinned commit
  `c560d38585b446bc93b0a62c77364bc8e7d266a4` (v1.15.0).
- Checkpoint structure inspected on CPU with `weights_only=True`.
  It initially rejected optimizer metric containers. Static pickle-opcode
  inspection found only tensor builders/storage and standard containers;
  explicitly allowing `collections.defaultdict`, `builtins.float`,
  `builtins.set` sufficed. No arbitrary pickle loading was enabled.
- Keys: model, optimizer, metrics, running_metrics, train_state,
  last_val_metrics, config, swa_model. Both raw and SWA parameters exist.
- Config: model version14,18nested bottleneck blocks,384trunk channels,
  Mish activation, metadata encoder with384internal channels. Do not
  confuse model version14 with engine/release version1.15.0.
- Upstream `Model.forward` accepts spatial, global and metadata inputs
  plus `ExtraOutputs`. It explicitly reports `trunkfinal` after final
  normalization/activation. Thus there is a source-supported path to a
  genuine spatial hidden representation, not just policy/value outputs.

## Subsequent verification

Strict raw/SWA network loading and actual trunk extraction passed. Raw
weights, not SWA, match the engine on the fixed policy checks. The8-game
benchmark supports numerical compatibility within specified tolerances,
not universal bitwise equivalence. GPU1memory and throughput were measured
and corrected after detecting a default-device statistics error. Extraction
completed23,211train/validation games. Conditional rank probes remain
negative; no production model or official submission changed.
