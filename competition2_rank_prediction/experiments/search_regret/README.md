# Actual-search regret

This branch tests information revealed by MCTS, not another neural-output
summary. Prior value extraction hardcoded maxVisits1; its depth3 meant
pipeline queue depth, not three-ply reading.

Protocols:[initial feasibility](PLAN.md),[fixed score pilot](PILOT.md).
No rank outcome is claimed until all8,000games and evaluation finish.

## Commands

From repository root, with `PYTHONPATH=competition2_rank_prediction` and
`OMP_NUM_THREADS=4`, use runtime/competition1_research/.venv/bin/python:

```bash
python -m competition2_rank_prediction.experiments.search_regret.probe
python -u -m competition2_rank_prediction.experiments.search_regret.extract
python -u -m competition2_rank_prediction.experiments.search_regret.evaluate
```

Probe output: production/search_regret_probe_20261109/probe.json.
Full output: production/search_regret_20261109, eight independent shards.
Atomic shard writes; source/model/config/cache/SGF hashes prohibit mixing
incompatible resumed runs. Evaluation requires EXTRACTION_COMPLETE and
refuses to overwrite existing reports. No official test arrays are used.

Each game supplies up to4own and4opponent surprising legal cached
decisions. Query both pre/post positions with32actual visits, not32-ply
depth. Save26descriptors, six per-decision values, eight BLACK-perspective
source values, pre/post/child visits, turns and masks. This lets an auditor
reconstruct both actor perspective and aggregate features independently.

Raw14controls are separate from search12features. Original frozen C1+C2
and same player-disjoint calibration are used; own/opponent/final-position
inputs unchanged. Baseline-cache name contains rank_joint_20261108, but
only original validation fields are read, not Joint or test predictions.
Missing values are imputed only from fitting-player games.

## Cost and early integrity checks

The16game capability probe returned192positions, all32visits; measured
159positions/s after warmup. The~13.4minute extrapolation for8,000games
is approximate and may underestimate harder selected middle-game positions.
GPU1 uses about2.1GiB in the actual extraction. GPU0 vLLM untouched.

First1,000completed games:8,000selected decisions, all pre/post visits32,
best-child visits min4,median16,max31. Own/opponent coverage100%. Independent
recomputation from saved source values verifies all four signed loss fields.
An additional first-game/eight-position-per-role check compared cached
surprisal to the actual actor's fresh policy: correlations0.99964own,
0.99761opponent. Mean absolute differences0.0875/0.1796nats reflect that
the original cache is not symmetry-identical; this is not bitwise equality
or a universal dataset-level equivalence claim.

The32visit root and best-child statistics have unequal effective budgets;
these losses are estimates, not exact game-theoretic regret. Protocol
controls and the two conditional score contrasts are essential.
Replay deliberately matches the frozen production convention (Japanese
6.5komi and legacy converted SGFs), not the opt-in handicap reconstruction.
Prior[handicap-fix findings](../../../runtime/competition2_research/experiments/handicap_fix/README.md)
document corrupted reconstruction/color semantics for some HD>=2games.
Thus this is a baseline-compatible search-feature pilot, not proof of
physically correct tactical regret on every original handicap game. No
mid-run reconstruction changes or retrospective score-based exclusions.

## Checks

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/search_regret
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/search_regret
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.search_regret.test_features
```

No pylint disables. Scoped mypy excludes imported legacy modules and
unavailable third-party stubs. No automatic submission, commit or push.
Current production Public-best0.7227214remains preserved.
