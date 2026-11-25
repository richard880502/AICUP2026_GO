# Unlabeled rank-prior adaptation screen

## Hypothesis

If each evaluation pool is designed with equal counts per rank, one can use
the unlabeled query probability matrix to fit a global per-rank offset that
makes its mean predicted distribution uniform. The existing submission uses
validation-fitted C1 calibration; this test checks whether unlabeled test-pool
adaptation helps after that calibration.

The equal-rank pool assumption is supported by the historical holdout3
generator (1,000 questions per cross-fit half, exactly100 per rank), but is
not established here as an official hidden-test guarantee. The experiment
must therefore be interpreted conditionally on that assumption.

## Fixed evaluation

- Use the already fixed NLL-C1 predictions from both holdout3 cross-fit halves.
- For each half, fit class offsets using only its unlabeled probability
  matrix and the existing uniform prior; do not use labels or ranks during
  fitting.
- Apply the offsets to every query in that half, then score once against
  labels for evaluation only.
- Report both halves, rank-count diagnostics and a paired-player95% interval.
- Advance only if utility gain is at least+0.003, interval lower bound>0,
  and both halves improve. No prior, temperature or subset sweep.
- Do not generate or alter an official submission.

This is a historically reused cohort and a conditional test of a known-prior
assumption, not an independent Public estimate.
