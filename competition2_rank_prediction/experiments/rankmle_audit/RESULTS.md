# rankmle premise audit: ten HumanSL profiles pass the gate, the rankmle rule does not

Completed 2026-11-18. Premise audit for [djma/rankmle](https://github.com/djma/rankmle), which queries KataGo HumanSL under
26 rank profiles and predicts the profile with the highest mean log-likelihood of the played moves. Production queries only
three HumanSL profiles (rank_5k, rank_1d, rank_5d); the one ingredient rankmle has that we lack is the extra profiles, so this
audit isolates exactly that. Validation cohort only; exploratory, not a production or Public result.

## Decision

The preregistered gate (`ten - three >= +0.003` with paired player-bootstrap 95% lower bound > 0) **passes**:
+0.014674 [+0.000680, +0.028837]. With the same simple learner, ten profiles carry some rank information that the three
production profiles do not. The lower bound is only just above zero.

The rankmle rule itself is far worse than a learner on our three profiles (-0.318414 raw, -0.180531 after removing per-profile
bias), and the surprisal channel alone over all ten profiles adds nothing (+0.000392 [-0.022421, +0.022534]). The gain therefore
does not come from the likelihood that rankmle uses; it comes from the log policy rank / entropy channels under the extra
profiles.

Consequence: the audit no longer rules out a ten-profile extraction. Whether it improves production is NOT shown here; that
needs the extra profiles fed to the production learners (Transformer/ResNet) on the full training corpus and an untouched
holdout before any adoption.

## Setup

- Cohort: `features/shards/val`, 8 shards x 1,000 games = 8,000 games, 400 players x 20 games, 40 players per rank.
- Features: `extract_rank_features.py`, 96 decisions per game, depth 3, streams `normal` + ten HumanSL profiles
  (rank_11k, 8k, 5k, 2k, 1d..6d, which map one-to-one onto D, C, B, A, 1D..6D). Per decision: surprisal (-log p of the played
  move), log1p(policy rank), policy entropy. Extraction: 6 workers on one GPU, 16:15-17:10 (first six shards ~2,440 s each,
  last two ~820 s each).
- Questions: 20 simulated questions per player, each 5-20 of the player's games drawn without replacement (fixed seed),
  8,000 questions in total. Question feature = mean over its games of the per-game channel means.
- Folds: 5 rank-stratified, player-disjoint folds; every learned arm is cross-fitted.
- Arms:

| Arm | What it is |
| --- | --- |
| `rankmle` | Faithful rule: per question, sum each game's mean log-likelihood under each of the ten profiles, take argmax. Untrained. |
| `rankmle_centered` | Same, after subtracting each profile's per-game mean log-likelihood on the training folds (ten parameters). |
| `three` | Standardized multinomial logistic regression (C=1.0), expected-utility decoding; streams normal + rank_5k/1d/5d, all 3 channels (12 features). |
| `ten` | Same learner; streams normal + all ten profiles, all 3 channels (33 features). |
| `ten_loglik` | Same learner; normal + ten profiles, surprisal channel only (11 features) - the information rankmle uses. |

- Metric: competition utility (1 exact, exp(-1) adjacent, 0 otherwise). Paired deltas are versus `three`, with a 5,000-draw
  player bootstrap 95% interval.

## Results

| Arm | Utility | Exact | >=2-rank error |
| --- | ---: | ---: | ---: |
| rankmle | 0.255781 | 15.65% | 57.36% |
| rankmle_centered | 0.393665 | 28.68% | 42.26% |
| three | 0.574196 | 42.13% | 16.30% |
| ten | 0.588870 | 44.25% | 15.96% |
| ten_loglik | 0.574588 | 41.53% | 15.16% |

Paired utility delta versus `three`:

| Arm | Delta | 95% interval |
| --- | ---: | --- |
| rankmle | -0.318414 | [-0.364484, -0.273355] |
| rankmle_centered | -0.180531 | [-0.218991, -0.141126] |
| ten | +0.014674 | [+0.000680, +0.028837] |
| ten_loglik | +0.000392 | [-0.022421, +0.022534] |

Redundancy with the `three` expected rank (Spearman over the 8,000 questions):

| Quantity | Spearman |
| --- | ---: |
| rankmle argmax | 0.9108 |
| rankmle_centered argmax | 0.8783 |
| ten expected rank | 0.9949 |

rankmle raw prediction histogram (true labels are uniform, 800 questions per class):

| D | C | B | A | 1D | 2D | 3D | 4D | 5D | 6D |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 31 | 325 | 2,157 | 3,486 | 1,461 | 468 | 59 | 1 | 10 | 2 |

The raw rule collapses onto B/A/1D (7,104 of 8,000 predictions); removing the per-profile bias raises utility from 0.255781 to 0.393665
but it stays far below the learned arms. Ordering is largely shared with the existing three-profile axis (Spearman 0.88-0.91),
consistent with `experiments/a1d_boundary` (1d-vs-2k ratio correlates 0.80 with the strength axis).

## What was and was not tested

Tested: whether seven additional HumanSL profiles add rank information beyond the production three, holding the learner fixed;
and how the faithful rankmle argmax rule compares on the same questions.

Not tested: the production ordinal Transformer, ResNet or opponent-identity evidence with ten profiles; fusion with the current
holdout3 pipeline (~0.764); the training corpus; rankmle's full 26 profiles; any Public submission.

## Caveats

- Simple learner, not the production Transformer. Absolute utilities (~0.57-0.59) are far below production and only the
  within-learner comparison is meaningful. The production Transformer already reads per-decision sequences of the three
  profiles, so it may capture part of what the extra profiles add here; the gain may shrink or vanish there.
- Validation cohort only (400 players). The 20 questions per player are overlapping subsets of the same 20 games, so the
  effective sample is the 400 players; the bootstrap resamples players accordingly.
- The lower bound (+0.000680) is barely above zero; this is one fixed seed / fold split and was not re-checked under others.
- `ten` has 33 features versus 12 for `three`; the comparison holds the learner fixed, not the feature count.
- 10 profiles, not rankmle's 26; intermediate profiles (e.g. 9k, 3k, 7d+) were not queried.
- The ten-versus-three prediction ordering is almost identical (Spearman 0.9949): the extra information is a small refinement
  of the same strength axis, not an independent signal. Registered rank only weakly reflects playing strength, which caps any
  strength-based evidence.

## Reproduction

`run.sh` (extraction on w77 GPU 1, then the CPU-only audit, ~30 s). Script: `audit.py`. Report:
`/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit/report.json`; features alongside in `feat10_val/` (not committed).
