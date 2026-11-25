# Direct semi-hard mining: positive versus NLL, uncertain versus random pairs

Completed 2026-11-08 after the user's explicit request to try mining.
The [direct preregistration](DIRECT_MINING_PREREGISTRATION.md) records the
new experiment following the prior gate failure. Prior reports and their
decisions are preserved; this experiment does not retroactively pass them.

## Main results

Three seeds, same architecture, player splits, fixed queries, 60 epochs x64
updates, pair weight0.05 and minimum-inner-NLL checkpoint selection.
The primary experimental addition is C's30% semi-hard /70% random pairs.

| Mean across seeds, evaluation300 players | A NLL | B random pairs | C mining |
| --- | ---: | ---: | ---: |
| Competition utility | 0.579449 | 0.589334 | 0.592667 |
| Exact accuracy | 44.33% | 45.44% | 45.78% |
| >=2-rank error | 18.67% | 17.89% | 17.56% |
| 1D/2D conditional AUC | 0.672593 | 0.668519 | 0.668148 |

| Paired comparison | Utility delta | 95% player CI | Adjusted97.5% CI |
| --- | ---: | --- | --- |
| C minus A | +0.013218 | [+0.001048, +0.025914] | [-0.000441, +0.027177] |
| **C minus B (primary mining effect)** | **+0.003333** | **[-0.004686, +0.012055]** | **[-0.005824, +0.013366]** |

All three seeds beat A; two beat B. C-B seed deltas are +0.000881,
+0.013333, -0.004214. C-A deltas are +0.012107, +0.002453, +0.025095.
Bootstrap uses5,000 PLAYER resamples after averaging each player's three
paired seed differences. The adjusted intervals account for these two
comparisons only, not the full history of exploratory hypotheses. Seed
uncertainty and repeated use of the cohort remain limitations.

The unadjusted C-A interval is positive, but C-B crosses zero. Therefore
the combination of pair supervision and mining has a positive exploratory
signal versus pure NLL; mining's incremental contribution over random
pair supervision is not established. The fixed further-study gate fails
because adjusted lower bounds are nonpositive and C's pair AUC is slightly
below B. No full-fusion/outer-confirmation study or production adoption
is triggered by this preregistration.

## Inner and secondary results

Inner utility means A/B/C are0.581470/0.577077/0.578367; inner C-A is
-0.003103 and C-B is +0.001290, both with intervals crossing zero. Inner
pair AUC changes0.752963/0.753704/0.754074. Epoch selection remains based
on inner NLL; no retrospective utility-based checkpoint selection occurs.

Predeclared probability ensembles score A/B/C:
0.578568/0.588568/0.592782 on evaluation. Ensemble C-B is+0.004214,
95% interval[-0.006667,+0.015095]; ensemble C-A is+0.014214,
interval[-0.000881,+0.031071]. These secondary results do not replace the
primary mean-seed analysis.

These scores describe a single own-player branch on300 exploratory
evaluation players. They are not official Public scores and cannot be
directly compared with full C1+C2 utility0.717012 on its2,000-player cohort.

## Mining and implementation verification

Each C arm finishes3,840 updates, with320 pure-NLL warmup updates and
3,520 auxiliary steps. It processes14,080 adjacent player pairs, with
28 strength refreshes (epochs6,8,...,60). Selection strengths use only
dense TRAINING players; all returned bags are recomputed with gradients.

| Seed | Mined | Random | Fallback | Selected C epoch |
| --- | ---: | ---: | ---: | ---: |
| 20261118 | 4,213 | 9,867 | 0 | 51 |
| 20261119 | 4,268 | 9,812 | 0 | 51 |
| 20261120 | 4,167 | 9,913 | 0 | 51 |

The mined share is29.94% overall. Recorded margins range from about
0.000054 to0.999956, strictly inside the specified(0,1] window. Each
auxiliary step uses four distinct low and four distinct high players.
The miner uses sorted scores and binary searches, not a full pair matrix.
Fallback was exercised by synthetic tests, although not needed in the run.

All C warmup metrics/losses match BOTH A and B exactly for each seed.
The fixed-control checkpoint SHA-256 hashes are unchanged. Checkpoint
normalization, seeds, budgets, inner-NLL selections, player order, query
labels and counts are verified. Per-player prediction arrays retain A/B/C
for independent score reconstruction. Unit tests cover training-only
refresh using held-out sentinel games, detached mining/mode restoration,
valid margins, inverted/zero/oversized-margin fallback, player uniqueness,
encoder gradients, seed aggregation and adjusted bootstrap confidence.

Local ignored artifacts:
`production/semi_hard_mining_direct_20261108/` contains protocol, split,
three checkpoints/histories, inner/evaluation A/B/C predictions, full
`report.json`, and `ABC_training_curves.png`. Code and reproduction commands
are in [README.md](README.md).
