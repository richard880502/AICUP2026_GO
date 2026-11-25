# HumanSL hidden representations — feasibility phase 2026-11-09

Objective: seek information beyond the policy/value summaries already
shown to be weak complements to C1+C2. No claim of a successful model yet.

Official HumanSL release offers a raw PyTorch checkpoint for experiments:
https://github.com/lightvector/KataGo/releases/tag/v1.15.0
Asset `b18c384nbt-humanv0.ckpt`,323,752,318bytes according to official API.

## Ordered feasibility gates before accuracy experiments

1. Download from verified official release URL only, into ignored runtime
   models directory. Record checksum and source release. Do not replace
   original `.bin.gz` or production models.
2. Obtain matching upstream model/board/feature code at pinned release
   commit. Inspect checkpoint structure and dependencies; no broad upgrades
   to the working venv. Source remains separately identifiable as vendor code.
3. Load on CPU first; GPU1 for inference. GPU0 vLLM never touched.
4. Reconstruct the exact input representation, including rules, komi,
   history, color and HumanSL metadata. Use fixed neutral1d profile for ALL
   examples initially, never supplied actual rank labels as model metadata.
5. On fixed held-out game positions, compare predicted played-move policy
   probability and full policy shape to the installed official engine,
   allowing documented precision/symmetry differences. Do not run rank
   experiments until equivalence is understood.
6. Verify extraction of a genuine hidden representation and stable,
   finite dimensions; measure GPU memory and games/sec on a small sample.
   Prefer pooled global plus played-point local features over saving every
   19×19channel tensor.
7. After feasibility, preregister an explicit player-disjoint conditional
   C1+C2 pilot, dimensionality reduction fitted on training players only,
   controls, uncertainty and advancement gate before inspecting outcomes.

Assets under runtime/competition2_research/models/human_trunk and
runtime/competition2_research/vendor/KataGo-human-v1.15.0. Runtime outputs
under production/human_trunk_20261109. Source/tests/records stay here.
No production replacement, test submission, commit or push without a
separate validated result/request. Python changes require pylint/mypy.
