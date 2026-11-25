# Exact-input and first-checkpoint audit, 2026-11-09

Read-only audit during the live matched training run. No experiment recipe
or hashed training source was changed.

- Training cache:4,000games/32,000valid positions. Evaluation:8,000games/
  64,000valid positions. All selected games have eight valid decisions.
- All96,000metadata rows were compared exactly against the fixed neutral1d
  metadata vector and matched. No sample's true rank appears through metadata.
- One representative raw SGF from each split was independently re-encoded:
  packed spatial planes, global inputs, metadata, played points and validity
  masks matched the stored cache bit-for-bit. This two-game replay is a spot
  check, not a universal independent replay of all feature construction.
- The first frozen arm finished40epochs and generated all8,000evaluation
  predictions. Probability shape/finiteness/normalization passed. Checkpoint
  selection matches the earliest minimum of the eight scheduled inner-NLL
  evaluations; optimizer/inner/evaluation player exclusion passed.
- No evaluation utility was computed during training. The first adapted arm
  was confirmed live at epoch5 after this check; that observation is historical,
  not proof of current process status.

Source/input hashes had independently remained unchanged at the preceding
audit. The added scoring module has its own source hash recorded at scoring
time; it was not part of the earlier training-source hash list.
