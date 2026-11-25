# End-to-end multi-game supervision

This branch asks whether pooling only at inference leaves useful training
signal unused. It does not add a new backbone or Set Transformer. Whole
question error backpropagates into the original decision-token encoder.

Fixed comparison: [PLAN.md](PLAN.md).
Completed outcome: [RESULTS.md](RESULTS.md). Pilot improves, full C1+C2
does not; not adopted.

- A: per-game NLL, equally weighted by player bag.
- B: half per-game NLL, half normalized log-mean whole-question NLL.
- Same 1,400 dense optimizer players, initialization, bag samples and
  augmentation; 300 disjoint inner players choose checkpoints, another300
  previously explored players provide the primary exploratory comparison.
- 3 seeds, 40epochs ×64steps, eight variable5/10/15/20-game bags per step.
- All ten ranks are supervised. This is not 1D/2D pairwise mining and does
  not operate on frozen embeddings.

Run from repository root with `PYTHONPATH=competition2_rank_prediction`
and `OMP_NUM_THREADS=4`:

```bash
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.question_supervision.run
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.question_supervision.evaluate
# Conditional on the pilot screen passing:
runtime/competition1_research/.venv/bin/python -u -m \
  competition2_rank_prediction.experiments.question_supervision.confirm
```

Default output: `production/question_supervision_20261109/`. Existing
artifacts are never overwritten. Source hashes, actual player splits,
data inventory, checkpoints, full inner histories and per-seed evaluation
predictions are retained. No production source/checkpoints/submission are
modified. GPU0's vLLM is untouched.

Validation:

```bash
runtime/competition1_research/.venv/bin/pylint \
  competition2_rank_prediction/experiments/question_supervision
runtime/competition1_research/.venv/bin/mypy \
  --follow-imports=silent --ignore-missing-imports \
  competition2_rank_prediction/experiments/question_supervision
runtime/competition1_research/.venv/bin/python -m unittest \
  competition2_rank_prediction.experiments.question_supervision.test_losses
```

The type check covers these changed modules, not imported legacy modules
or absent dependency stubs. The loss tests cover production pooling,
the inference probability floor, encoder-reaching gradients, bag-balanced
control weights and invalid bags. Screening requires a meaningful paired
gain before any full C1+C2 comparison, not just a better training score.
