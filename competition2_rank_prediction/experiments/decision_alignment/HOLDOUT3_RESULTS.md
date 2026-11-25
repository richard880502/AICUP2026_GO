# Decision-aligned calibration on current C1+C2: no confirmation

The fixed soft decision objective was evaluated on the historical 2,000-player
cross-fitted cohort. Each scored half was excluded from upstream model
training; the separate400-player validation cohort alone fitted fusion and
C1. The cohort was used in earlier C1/C2 research, so this is a
method-specific confirmation rather than a pristine benchmark.

| C1+C2 residual objective | Official utility | Exact accuracy |
| --- | ---: | ---: |
| NLL residual | 0.71701175 | 58.20% |
| Soft decision-aligned residual | 0.71637963 | 58.10% |

Decision-aligned minus NLL is `-0.00063212`, with paired-player95% interval
`[-0.00321242, +0.00194818]`. Half deltas are`-0.00352848`and`+0.00226424`.
The NLL baseline exactly reproduces the historical C1+C2 report. Keep NLL C1.

The prior borrowed-opponent experiment is a separate, weaker branch and its
positive calibration diagnostic did not transfer to current C1+C2. It is not
used as evidence for the production pipeline.

No temperature search, official inference, submission, upload, commit or push
was performed. The pre-existing production CSV is unchanged.
