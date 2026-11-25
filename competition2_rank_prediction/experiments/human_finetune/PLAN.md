# Matched HumanSL adaptation pilot, 2026-11-09

Fixed before rank training/evaluation. Feasibility synthetic gradients are
documented separately and do not select this recipe against rank utility.

## Hypothesis and controls

Frozen HumanSL trunk probes did not improve full C1+C2. Test whether updating
the final two pretrained residual blocks produces useful rank information,
rather than adding capacity only after fixed features. No architecture or
learning-rate sweep is allowed on the evaluation cohort.

- Reuse fixed500-player source manifests:400optimizer and100inner players
  from the coverage experiment's deterministic rank-stratified partition.
  Eight games each, eight own pre-move positions/game. All original400
  evaluation players remain disjoint. Historically explored, not fresh data.
- Same exact packed board inputs in both arms; fixed neutral1d metadata,
  Japanese rules and legacy baseline handicap convention. No rank metadata,
  official-test inputs or other-task candidate libraries.
- Mean global plus played-point trunk features, mean across valid positions
  within each game. Padding repeats a valid position but is excluded from the
  pooling denominator. Pass local features are zero. No temporal Transformer.
- Head:LayerNorm768 plus Linear768→10, identical initialization per seed.
  A freezes the complete backbone; B updates blocks16/17 and final trunk norm
  parameters. Both keep backbone.eval() to freeze running normalization stats.
- AdamW, headlr0.001, adapted backbonelr0.00001, weight_decay0.01,
  cosine schedule. Forty epochs×64updates, batch4games×8positions.
  Player-balanced sampling with identical player/game draws across arms.
- Seeds0/1/2. Select earliest minimum inner per-game NLL among fixed epochs
  5/10/15/20/25/30/35/40. Same inner800games for both arms. No evaluation
  score during training and no retrospective checkpoint selection.
- Save selected trainable weights, original checkpoint reference/hash,
  source/input hashes, optimizer/inner IDs, loss histories and all selected
  checkpoint metadata. Never modify official weights or production models.

## Scoring and advancement

Predict the original8,000evaluation games, pool exactly the historical fixed
whole-question samples, and use official expected-utility decoding. Primary
is three-seed probability-ensemble B−A. Report paired-player95% bootstrap,
individual seed deltas, per-rank and query-size results, plus learning curves.
Advance only for delta>=0.005, lower bound>0,2/3positive seeds and worst-rank
loss>=-0.05. Then test B as an additional member of complete original C1+C2,
with all fusion/C1 fitting player-fold-safe and original baseline replay.
Passing a single-branch pilot never authorizes production replacement.

## Resources and status

GPU1 only; GPU0's unrelated service must stay untouched. Six CPU workers build
isolated packed input caches under`production/human_finetune_inputs_20261109/`.
Missing COMPLETE marker must block training.
No automatic upload, commit or push. The initial GPU smoke suggests modest
memory use, but whole-run runtime and memory still need measurement.

## Launch record

Exact input caches completed:4,000training and8,000evaluation games,24shards,
approximately9.4MiB compressed. Binary packing round trips are checked per
game. Loader audit confirms expected counts and player exclusion.

`train.py` passes pylint10/10 and mypy. A real-rank one-step GPU test verified
bit-identical initial predictions and identical loss in both arms, unchanged
frozen backbone and changed adapted last-block parameter. Smoke weights were
discarded before launching actual fits.

All six fixed runs launched in exec session60895, writing only to
`production/human_finetune_20261109/`. Initial live observation:GPU1about
1,046MiB and99%utilization; GPU0serviceunchanged. This historical session ID
does not by itself prove ongoing execution; poll it before status claims.
Input caches, source files and the original pretrained checkpoint are hashed
in `input_hashes.json`. Do not change hashed Python sources while runs are
active. There is no scored adaptation result yet.

## Completion

All six runs and the conditional full-pipeline comparison have now completed.
The launch observations above remain historical. See [RESULTS.md](RESULTS.md)
for the final paired scores and failed production-adoption decision.
