# End-to-end question supervision — preregistration 2026-11-09

## Hypothesis and distinction from previous work

Production trains a per-game classifier and pools at inference. Dense bag
experiments train downstream networks on frozen game embeddings. Pair mining
adds scalar 1D/2D ordering. None of these directly trains the original
decision encoder with a normalized log-mean **whole-question NLL** across
all ten ranks. This is the proposed new experiment, not a claim of novelty
outside this repository or a proven improvement.

## Matched pilot

- Reuse existing decision features, not new KataGo extraction. Original
  own-side d96 architecture; no Set Transformer, Joint or extra parameters.
- Reuse the documented semi_hard_mining player partitions: dense train1400,
  inner300, evaluation300; sparse players excluded from this matched pilot.
  These cohorts have been explored before; evidence is exploratory.
- Fit normalization only on optimizer players, never inner/evaluation.
- Each optimizer batch samples8 distinct dense training players, with one
  no-replacement5/10/15/20-game query each. N schedule is shared across arms.
  Every game in a bag has the same player and true rank.
- A: per-game NLL, averaged within each bag then across bags.
- B: 0.5 × A +0.5 × normalized-log-mean question NLL. Gradient must flow
  through pooling and the decision encoder, no detached embeddings.
- Identical game draws, augmentations, initialization, architecture and
  optimizer steps. Thus B cannot win just by seeing more dense games.
- Seeds0/1/2,40epochs ×64updates; AdamW lr0.0005, cosine schedule,
  weight decay0.05, batch tokens noise0.05, token dropout0.15.
- Checkpoint selection: maximum inner official question utility, earliest
  wins ties; fixed questions, no evaluation/official-test checkpoint choice.
- Primary: paired seed-mean reward on the300 evaluation players; average
  seeds and repeated questions within player before bootstrap. Report
  95% interval, per-rank, per-query-size, NLL and seed consistency.
- Screen to full C1+C2 only for gain>=0.005, positive interval lower bound,
  >=2/3 positive seeds and no rank decline>0.05. No new submission merely
  because a single score rises. The Public0.7227214 original remains frozen.
- Full-pipeline confirmation must use neural-unseen players for every
  branch; do not evaluate a newly trained encoder on its training players
  while calling a cross-fit calibration layer leak-free.

## Resources and deliverables

Other-session soft-boundary training is live on GPU1. Never kill/restart it;
GPU0 vLLM is also out of scope. CPU preflight, tests and implementation may
proceed while waiting. Launch substantial new GPU training only after a
fresh process/resource check establishes sufficient capacity.

Source/tests here; all artifacts under production/question_supervision_20261109.
Keep an input/split audit, full histories, selected checkpoints, paired
results, learning curves and written interpretation. Run pylint and mypy
after Python changes. No automatic commit/push or Aidea upload.
