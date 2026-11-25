# Three post-processing changes to the v47 opponent model — all rejected

Completed 2026-11-17. Exploratory holdout3 cross-fit in the same two-half
protocol as `h3_final_v47.py`, reading only the cached `h3_sims_v47.pt`. No
re-extraction, no official inference, no file under `production/` touched.

The baseline replays the v47 holdout3 number: 0.7644 here against 0.7632
recorded for v47, and 0.7644 / 0.7647 / 0.7639 across three seeds, so roughly
±0.0005 of refit noise. C1+C2 replays 0.7170 exactly.

## Results (3 seeds, cross-fitted, paired question bootstrap)

| Variant | Score | Delta | 95% CI | Halves |
| --- | ---: | ---: | --- | --- |
| v47 baseline | 0.7644 | — | — | — |
| Confidence-dependent unknown share | 0.7620 | -0.0023 | [-0.0088, +0.0042] | -0.000 / -0.005 |
| Per-question-size fusion weights | 0.7608 | -0.0035 | [-0.0086, +0.0015] | -0.001 / -0.006 |
| Both | 0.7570 | -0.0073 | [-0.0146, -0.0002] | +0.003 / -0.018 |
| CSLS, beta 0.25 | 0.7553 | -0.0094 | [-0.0153, -0.0036] | -0.009 / -0.010 |
| CSLS, beta 0.50 | 0.7520 | -0.0127 | [-0.0199, -0.0055] | -0.010 / -0.016 |
| CSLS, beta 0.75 | 0.7486 | -0.0161 | [-0.0245, -0.0077] | -0.013 / -0.019 |
| CSLS, beta 1.00 | 0.7387 | -0.0260 | [-0.0359, -0.0160] | -0.024 / -0.028 |
| CSLS, beta 1.50 | 0.7230 | -0.0417 | [-0.0542, -0.0294] | -0.037 / -0.047 |

## 1. Per-question-size fusion weights

Motivated by the main record's line "Same, weights per question size 0.7646",
+0.0061 over the flat 0.7585. That measurement was taken on the **v4-only**
model. v47 (the v4+v7 cosine average) already reaches 0.7632-0.7647 on its
own, and per-size weighting on v4 alone reached 0.7646 — the same place. The
two changes capture the same gain, and stacking them only adds parameters:
`log_alpha`, `gamma` and `base_weight` become per-bucket (four buckets split
at 8/11/14 games) and the result regresses.

## 2. Confidence-dependent unknown share

`self.unknown` depends only on the game kind, never on how well the opponent
was actually identified. Replacing `sigmoid(unknown[kind])` with
`sigmoid(unknown[kind] + w * margin)`, where margin is the standardised gap
between the top class cosine and the next ten, regresses. Related negative:
`experiments/dynamic_temperature` (calibration improves sharply, utility does
not).

## 3. CSLS

The strongest negative of the three, and monotone in beta: every increment of
the hubness correction costs more utility, with intervals clear of zero.

CSLS is effective in Competition 1, so this was expected to transfer. It does
not, and the reason is structural. In Competition 1 each candidate is the
answer to at most one query, so a class that is similar to everything is pure
nuisance and subtracting its hubness term is free. Here the posterior is
pushed through `@ ranks` into a distribution over the opponent's registered
rank, and a player who looks similar to many games is often genuinely a more
frequent opponent. The hubness term partly encodes a real prior over who the
opponent is, so removing it discards signal.

Implementation note: with a per-game softmax, the CSLS per-game term is
constant across classes and cancels, so the correction reduces to subtracting
`beta * r(c)` from the class cosines; `r(c)` is the mean of each class's top-10
similarities over the query games, and beta was swept rather than learned.

## Conclusion

None of the three is adopted. Taken with
[strength_score_v2](../strength_score_v2/RESULTS.md) and
[full_train_coverage](../full_train_coverage/RESULTS.md), the error
decomposition explains why: against the v47 baseline about 33% of questions
are off by exactly one rank and only about 2.3% by two or more, so gross
errors are nearly exhausted and only adjacent-rank discrimination is left.

Measured on the same cached inputs, even the most confidently identified
opponents leave that resolution out of reach:

| Subset | P(opponent rank = own) | P(within 1) | SD |
| --- | ---: | ---: | ---: |
| All games | 0.318 | 0.690 | 1.74 |
| Top 50% confident | 0.394 | 0.775 | 1.47 |
| Top 20% confident | 0.477 | 0.861 | 1.19 |

Matchmaking pairs players to about ±1.2 ranks even when identification is
effectively certain, which is the same scale as the remaining error.

## Checked and ruled out while reading the model

- The transition table clamps the rank difference to ±3. Only 1.8% to 5.6% of
  games exceed that, so the clamp is not losing usable evidence.
- Games with handicap >= 2 are discarded by `usable = kind < 3`: 1,204 games
  (4.85%), affecting 413 questions, with 14 questions losing every game. The
  handicap size directly constrains the rank difference, so this is the one
  structural signal still unused. Not attempted here.
