# Shared unknown opponents — audit, not built

Completed 2026-11-18 with `audit_shared_opponents.py`. Read-only against the
cached official inputs; nothing was trained and no submission was produced.

## Why this was run first

[RESULTS.md](RESULTS.md) lists one remaining candidate:

> A joint model over unknown shared opponents was estimated at +0.002 to +0.005
> and not built.

Two experiments immediately before this one failed on unchecked premises rather
than on execution — [strength_score_v2](../strength_score_v2/RESULTS.md) measured
the same latent variable the pipeline already had, and
[rank_context](../rank_context/RESULTS.md) recovered a handicap channel that
turns out to be empty in the official test set. So the premise was tested
against the official inputs before any model was written.

## Decision

**Do not build it.** The detector that demonstrably finds 447 repeated
opponents among identifiable games finds no excess at all among unidentifiable
ones, and every repeat it does find belongs to an opponent whose registered
rank v47 already uses.

## 1. Known and unknown cannot be separated per game

| | median top cosine |
| --- | ---: |
| absent control (players v4 never saw) | 0.5003 |
| official opponent games | 0.5140 |
| present control (unseen games of present players) | 0.5259 |

At the 99th-percentile absent cut (0.6549) the mixture estimate is

```text
(0.0647 - 0.0100) / (0.0823 - 0.0100) = 0.7568
```

which reproduces the recorded 0.69-0.77 known share. But that is a **population
mixture, not a per-game label**: only 8.2% of genuinely present games clear the
cut, and just 556 of 10,000 official games are confidently identified. A joint
model restricted to "unknown" games cannot be told which games those are.

## 2. Cross-game similarity is a valid same-player detector

Per-game representation is the v4 style embedding and the v7 class-cosine
profile, each L2-normalised, averaged with equal weight. The v7 profile is
mean-centred first so the shared "similar to everyone" component drops out.

| Cross-question pairs, both games confidently identified | mean | q999 |
| --- | ---: | ---: |
| top class agrees (same player), n=447 | **0.7162** | - |
| top class differs (different players), n=153,547 | **0.0659** | 0.6201 |

At the 0.1% false-positive cut of 0.6201, 90.4% of same-player pairs are
detected. The separation is wide and the detector works.

### Same-question pairs are not a valid null

| | mean |
| --- | ---: |
| same-question pairs | 0.1795 |
| cross-question pairs | 0.0296 |

The opponents within one question are nearly always different people, yet they
are six times more similar to each other than random cross-question pairs. The
opponent-side embedding therefore carries **question-level context** - the asked
player's own style and rank neighbourhood - not opponent identity alone. A first
pass of this audit used same-question pairs as the different-player null and
every result inverted. Any future opponent-side analysis should avoid that null.

## 3. All detectable recurrence sits in already-identifiable games

| Threshold (detection) | Stratum | Pairs | Observed | Expected | Excess | Implied repeats |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.6201 (90.4%) | both confident | 153,994 | 558 | 154.0 | **+404** | **447** |
| | neither confident | 44,527,264 | 36,750 | 44,527.3 | **-7,777** | **0** |
| 0.7166 (53.9%) | both confident | 153,994 | 257 | 15.4 | **+242** | **448** |
| | neither confident | 44,527,264 | 2,198 | 4,452.7 | **-2,255** | **0** |

Two thresholds whose detection rates differ by 1.7x imply **447 and 448**
repeat pairs respectively. That agreement is the calibration check: the
estimator is consistent, not lucky.

Those repeats are all among confidently identified opponents - 333 distinct
opponents, 108 of them in two or more questions (32.4%) - and v47 already uses
their registered rank, so they are exactly the recurrence the joint model does
**not** need.

## 4. Detection power, honestly

| Cross-question pairs sharing a top class | mean | above 0.6201 |
| --- | ---: | ---: |
| both confident | 0.7162 | **90.4%** |
| one confident | 0.5831 | 40.2% |
| neither confident | 0.4859 | **14.4%** |

Power on unidentifiable games falls roughly six-fold, so this is **no positive
evidence** rather than proof of absence. Two things still argue against the
premise:

- Scaling the identifiable set's recurrence (447 pairs per 556 games) to the
  9,444 unidentifiable games would predict roughly 7,600 same-player pairs.
  Even at 14.4% power that is about 1,100 detections above the null. The
  observed excess is **negative** at both thresholds.
- Forcing links at the strict cut produces 379 clusters, the largest spanning
  **232 questions**. A single opponent shared by 232 of 800 test players is not
  a plausible reading; it is transitive-closure noise.

The 14.4% figure is itself an upper bound on degradation, since top-class
agreement is unreliable for exactly those games.

## 5. The idea is weak even if the premise held

What a joint model adds is second-order: that two players of unknown rank faced
the same opponent of unknown rank. The existing path - identify the opponent,
read their registered rank - is first-order and already in production. The
+0.002 to +0.005 estimate in RESULTS.md was never measured, and the 982/361
figure it was reasoned from counts **identified** opponents, whose ranks are
already used.

## Reproduce

```bash
PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
  competition2_rank_prediction/c1/.venv/bin/python -m \
  competition2_rank_prediction.experiments.opponent_identity.audit_shared_opponents
```

Needs `test_opp_scores.npz`, `test_opp_sims_v7.npy` and `controls.npz` under
`/tmp2/richard/claude_scratch/opp_probe`, plus the `style_v4` checkpoint. GPU 0
hosts a vLLM service with too little free memory for the 10,000 x 10,000
similarity matrix; use GPU 1.
