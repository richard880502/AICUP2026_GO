# Actual-search regret — next feasibility direction, 2026-11-09

## Evidence for a distinct gap

The legacy extract_value_features.py calls make_analysis_query(...,1,...).
Its depth flag controls pipeline batches in flight, not tactical search
depth. Policy/HumanSL/trunk outputs mostly describe neural first instinct;
their negative results do not test regret revealed by actual MCTS search.

This is a new direction AFTER both frozen-trunk probes failed; not a
post-hoc claim that their gates passed. Original Public-best C1+C2 frozen.

## Before any accuracy outcomes

1. Benchmark normal b18model, no human model, GPU1only; maxVisits32 is
   visit budget, NOT a fixed32-ply depth. Query sampled pre/post actual-move
   positions; inspect raw root fields, searched root values and best-child
   values. Record throughput/visits and numerical sanity before scaling.
2. Candidate pilot position selection: up to4own+4opponent decisions per
   game, highest cached normal-policy surprisal among legal/masked96decision
   samples. Prefer move indices20..min(250,last−10); deterministic fallback
   if insufficient. Selection uses no rank labels. Freeze exact rule in
   source and manifest before any rank-score evaluation.
3. Derive raw and searched score/winrate losses with BLACK-to-player
   perspective, plus their difference. Keep raw-quality controls; do not
   attribute gains from selecting unusual moves to search itself.
4. Use original400-player/8,000-game validation cohort only for first pilot;
   original frozen C1+C2 members and player-disjoint calibration. No new
   test predictions. Pre-register compact exact features, models and
   statistical gate after API/throughput feasibility, before scores.
5. If projected cost is excessive, report measured cost and select a
   smaller predeclared feasibility scope, not silent premature results.
   Do not switch budget based on rank outcomes.

Existing controls imply any credible breakthrough must demonstrate
conditional full-pipeline information, not just a strong standalone branch.
The indexed dataset has no wholly unused player IDs beyond original
train/validation; fresh-data claims require caution and retraining splits.

Source/tests here, runtime outputs under production/search_regret_20261109.
No production overwrite, automatic submission, commit or push. GPU0's
vLLM untouched. Goal remains active until an actual breakthrough is found
or the user changes the task; this plan alone does not fulfill it.
