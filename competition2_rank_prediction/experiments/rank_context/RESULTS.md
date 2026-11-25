# Rank Context v1 — the handicap channel does not exist in the official test set

Completed 2026-11-18. Exploratory holdout3 cross-fit, not a Public result and
not a new submission.

## Decision

**Do not advance, and do not pursue the indicated follow-up either.** The
handicap branch works on holdout3 and is structurally safe, but the official
test set contains **no HD>=2 game at all**, so the channel it recovers is
empty where it would have to pay off:

```text
official opponent-side games   10,000 over 800 questions
handicap histogram             {0: 7720, 1: 2280}
questions with any HD>=2       0
questions with every game HD>=2  0
```

`link_test.py:handicaps()` parses `HD[...]` with no fallback, so it would
raise rather than silently return zero; it finds 2,280 HD1 games in the same
pass. The absence is real, not an extraction artifact.

holdout3 is therefore **not distributionally identical to the official test
set**: it carries 1,204 HD>=2 games (4.85%) that the official questions do
not. Any future holdout3 measurement that leans on handicap structure should
be discounted accordingly.

## Scores (2,000 holdout3 questions, cross-fitted)

Baseline refits per run; three runs in this session gave 0.764038 / 0.765302 /
0.763670, so roughly +-0.0008 of refit noise. Only paired deltas are read.

| Variant | Delta | 95% CI | Halves | no HD>=2 (1,587) | all HD>=2 (14) |
| --- | ---: | --- | --- | ---: | ---: |
| `handicap` | +0.00047 | [-0.0028, +0.0045] | +0.0017 / -0.0008 | 0.00000 | +0.2143 |
| `context` | -0.00456 | [-0.0135, +0.0009] | -0.0043 / -0.0048 | -0.00126 | +0.2594 |
| `both` | -0.00801 | [-0.0170, -0.0015] | -0.0074 / -0.0086 | -0.00474 | +0.1880 |
| `handicap_alone` | +0.00063 | [-0.0012, +0.0020] | +0.0013 / 0.0000 | 0.00000 | +0.0903 |
| `handicap_decay` | +0.00074 | [-0.0024, +0.0031] | +0.0011 / +0.0004 | 0.00000 | +0.1355 |
| **`handicap_fit_global_apply_alone`** | **+0.00150** | [-0.00018, +0.00337] | +0.0023 / +0.0007 | 0.00000 | **+0.2143** |

Gate is `delta >= +0.003`, CI lower bound `> 0`, both halves `>= 0`. The best
variant passes only the third.

## What the experiment did establish

The handicap evidence is **valuable only where it is the sole evidence, and
dilutive everywhere else.** In the as-designed `handicap` variant the 14
questions whose every game is HD>=2 gain +0.214, while the other 399 questions
that merely contain an HD>=2 game lose about 0.009 each and cancel it out. The
learned uncertainty explains why: `hd_sigma` settles near 1.96-2.04 ranks,
far wider than even-game evidence, yet the per-game log-evidence is summed as
though it were of comparable quality.

The learned transition is otherwise exactly what matchmaking should produce,
monotone in handicap size and correctly signed:

```text
own gives     HD2 +0.49  HD3 +0.96  HD4 +1.43  HD5 +1.90  HD6 +2.37
own receives  HD2 -1.21  HD3 -1.95  HD4 -2.68  HD5 -3.42  HD6 -4.15
```

### Added variants

`handicap_alone` applies the branch only to questions with no HD0/HD1 game.
It is provably inert elsewhere (no HD>=2 subset delta is exactly 0.00000), but
it scores only +0.090 on the 14 questions against +0.214 for the global fit:
fitting under that restriction leaves the branch touching about seven
questions per training half, far too few for its nine parameters.

`handicap_fit_global_apply_alone` separates the two. The transition is a
property of the matchmaking system, so it is estimated from all 1,204 HD>=2
games; only the *application* is restricted. This recovers the full +0.214 on
the 14 questions with zero change on the other 1,986, and is the correct
ceiling for the idea: 14/2000 x 0.2143 = +0.0015, which the measured overall
delta reproduces exactly. Twelve answers change in total.

That ceiling is half the gate, and it is zero on the official set.

## Fix applied

`build_context.py` read the handicap from the holdout3 question CSVs, whose
SGFs are stripped to `SZ` plus the move list and carry no `HD` property, so
every game parsed as handicap 0. Its own consistency check caught this
(6,013 games disagreed with the cached v47 kind - exactly the count of all
non-even games). The handicap is now read the way `h3_dump.py` reads it: the
authorized training CSVs, keyed by the global row index in `h3_map.npz`. The
check then passes, and the recovered counts match the v47 cache: 1,204 HD>=2
games, 413 affected questions, 14 fully affected.

## Reproduce

```bash
GPU=1 DEVICE=cuda:0 bash competition2_rank_prediction/experiments/rank_context/run.sh
```

GPU 0 hosts a vLLM service with about 4.6 GB free, which is not enough to hold
the 24,822 x 6,807 score matrix and its softmax graph; use GPU 1.
