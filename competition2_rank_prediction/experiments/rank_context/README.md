# Rank Context v1

## Why this experiment exists

Strength Score v2 showed that a completely different own-move representation
and objective still learned almost the same latent variable as the production
pipeline:

```text
corr(Strength Score, baseline expected rank) ~= 0.96
fusion weight = 0 on both folds
```

Full-player training coverage also failed. The next hypothesis is therefore
not "estimate playing strength better", but:

> estimate the residual between playing strength and the **registered rank**
> by using structural matchmaking context.

The v47 opponent model currently drops every game with `HD >= 2`:

```python
usable = kind < 3
```

That is 1,204 holdout3 games affecting roughly 413 questions, including a
small number of questions that lose all opponent evidence. Exact handicap
size and which side receives the handicap are direct server/matchmaking
signals, not another own-move strength estimate.

## Branch

```text
experiment/rank-context-v1
```

## Two independent additions

### A. Exact handicap-aware opponent transition

The original v47 path is left unchanged for:

- even games
- HD1, opponent White
- HD1, opponent Black

For `HD >= 2`, instead of discarding the game, the model learns:

```text
P(own registered rank | opponent registered rank,
                         exact handicap,
                         own gives/receives handicap)
```

The transition is deliberately low-parameter. Let:

```text
d = own_rank - opponent_rank
```

For each direction (own gives / own receives), the center is:

```text
|center(h)| = softplus(a) + softplus(b) * (h - 2)
```

with sign determined by who receives handicap. A learned sigma controls how
uncertain the rank gap is. A learned gate can drive HD>=2 evidence toward zero
if the signal is not useful.

This adds only a handful of parameters.

### B. Question-level structural residual head

A zero-initialized linear residual head sees only structural features:

- question size
- own Black fraction
- HD1 fraction
- HD>=2 fraction
- receive/give fractions
- mean/max handicap
- signed mean handicap
- HD2 / HD3 / HD4+ receive/give histograms

It does **not** inspect the player's own moves.

The residual head adds rank logits on top of the frozen v47 fit.

## Important protocol detail

For each target fold:

1. fit the original v47 model on the opposite fold;
2. freeze **all original v47 parameters**;
3. initialize each new variant from exactly that fitted v47 state;
4. optimize only the newly introduced parameters;
5. evaluate on the untouched target fold.

This avoids mistaking "v47 got 200 more Adam steps" for a real gain.

The four reported models are:

```text
baseline  = exact v47
handicap  = baseline + exact HD>=2 transition
context   = baseline + structural residual head
both      = baseline + both additions
```

## Run

From the repository root:

```bash
git fetch origin
git checkout experiment/rank-context-v1

GPU=0 DEVICE=cuda:0 bash competition2_rank_prediction/experiments/rank_context/run.sh
```

Defaults reuse the existing artifacts:

```text
/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt
/tmp2/richard/claude_scratch/opp_probe/h3_opp.csv
runtime/competition2_research/features/holdout3/
```

Override if needed:

```bash
STATE=/path/to/h3_sims_v47.pt H3_OPP=/path/to/h3_opp.csv GPU=0 DEVICE=cuda:0 bash competition2_rank_prediction/experiments/rank_context/run.sh
```

No HumanSL extraction and no large-model training is required.

## Output

```text
competition2_rank_prediction/experiments/rank_context/results/report.json
```

The report contains overall and fold deltas, paired bootstrap CI, plus
separate results for:

- questions containing at least one HD>=2 game
- questions containing no HD>=2 game
- questions whose every game is HD>=2

It also records the learned handicap rank-gap centers, uncertainty, evidence
gate, unknown share, and structural-feature coefficient magnitudes.

## Advancement gate

The default gate is intentionally stricter than "positive":

```text
delta >= +0.003
paired 95% CI lower bound > 0
both fold deltas >= 0
```

If the overall result is near zero but the HD>=2 subset is strongly positive,
that is still useful evidence: the next experiment would conditionally apply
the handicap branch only to affected questions rather than force a global
change.
