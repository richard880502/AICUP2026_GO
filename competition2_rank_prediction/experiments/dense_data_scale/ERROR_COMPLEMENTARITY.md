# Shared-error diagnostic, 2026-11-09

Read-only comparison of eight completed full-pipeline candidates on the same
8,000 simulated questions /400historically explored players. Labels/player
order and the entire original baseline probability array matched exactly
across caches (absolute tolerance1e-12). No router or new model was trained.

| Candidate | Decisions changed | Questions improved | Questions worsened |
| --- | ---: | ---: | ---: |
| Reward-aligned encoder | 261 | 131 | 103 |
| Question-supervised encoder | 382 | 166 | 177 |
| HumanSL output context | 1171 | 567 | 559 |
| Frozen trunk linear probe | 101 | 56 | 42 |
| Frozen trunk temporal probe | 78 | 40 | 38 |
| Search-regret features | 662 | 300 | 344 |
| Adapted HumanSL trunk | 55 | 23 | 24 |
| Eight-game coverage encoder | 466 | 188 | 240 |

Changed decisions can have identical utility, so improved+worsened need not
sum to changed. Comparison uses official exact1/adjacentexp(-1)/otherwise0.

An **answer-informed per-question oracle** choosing the best of the original
and eight candidates scores0.7720620132 versus original0.7106057000, apparent
headroom+0.06145631. At least one candidate improves820questions;6,306have
identical reward across all methods;314receive zero reward under every method.
This oracle is not deployable, not a learned selector result, not a statistical
upper bound on unseen data, and not evidence that0.77is achievable. Many
methods were explored on this same cohort; oracle selection is optimistic.

## Implications

The observed failures do not establish that every new method makes exactly
the same mistakes. Some complementary decisions exist, but gains and losses
largely cancel. The diagnostic does not establish a label-free rule that
separates helpful from harmful changes.

Do not fit a router to globally cached OOF outputs and call another random
split independent: a training row's upstream calibration may have used a
final-held-out player's label. A learned second-stage selector requires
nested player-safe calibration, not merely player-grouping its final layer.

A cheaper distinct next hypothesis is two-degree-of-freedom, score-aware
relative weighting of the original frozen members **after accounting for the
fixed C1 correction**. Existing production fusion minimizes per-game NLL;
the old question_fusion experiment minimized question NLL and added class
biases. A tightly bounded utility-surrogate relative-weight fit is different
and avoids a high-capacity nine-method router. It still requires held-player
evaluation and matched NLL controls before any adoption claim.
