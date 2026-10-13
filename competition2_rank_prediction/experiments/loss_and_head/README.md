# Output head and loss comparison (2026-10-11) - rejected

Dense 96-decision features, single seed, validation cohort. Variants of `transformer_rank.py --head {clm,softmax} --loss {nll,soft,escore} --select {nll,score}`.

| variant | per-game top-1 | question, mean + argmax | mean + expected | log-mean + expected |
| --- | --- | --- | --- | --- |
| ordinal head + NLL (production) | 0.3056 | 0.489 | 0.559 | 0.587 |
| softmax + NLL | 0.3091 | 0.492 | 0.563 | 0.590 |
| softmax + scoring-rule soft labels (1 / e^-1) | 0.3098 | 0.522 | 0.581 | 0.587 |
| softmax + maximise expected score | 0.3068 | 0.476 | 0.522 | 0.566 |
| ordinal head + soft labels | 0.3089 | 0.515 | 0.574 | 0.586 |

Soft labels help the simple decoders but the production log-mean + expected-score decoder already gets the same effect; best scores 0.586-0.590 are within noise. Rejected.
Artifacts: `results/` (per-variant probabilities and checkpoints), `logs/`.
