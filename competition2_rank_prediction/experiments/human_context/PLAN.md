# HumanSL conditional value/uncertainty — 2026-11-09

## Motivation and verified capability

Current production stores only surprisal/rank/entropy for normal and three
HumanSL profiles. Prior normal-KataGo value/ownership experiments did not
establish a useful complement. This branch instead tests **human-model**
value and uncertainty conditioned on5k/1d/5d, plus actual-move changes.

Official field definitions:
https://github.com/lightvector/KataGo/blob/master/docs/Analysis_Engine.md

A single-game local capability probe found all five fields. Average
across-profile ranges: winrate0.08653, score mean3.20748, score stdev8.30537,
short-term winrate error0.03248, short-term score error1.01836.
This is not evidence of rank accuracy. It verifies only a working interface.
Probe output is production/human_context_probe_20261109/probe.json.

## Fixed pilot, before any rank-score evaluation

- Original400-player/8,000-game validation cohort only. No official test
  extraction, model download, trunk modifications or production changes.
-16evenly spaced own decisions; query both before and after the actual
  move; maxVisits1, deterministic symmetry; HumanSL5k/1d/5d.
- Fields: humanWinrate, humanScoreMean, humanScoreStdev, humanStWrError,
  humanStScoreError. Normal analogues collected as a matched control.
- Winrate/score converted from configured BLACK perspective to the labelled
  player's perspective. Scores/stdev/errors in points divided by20.
- Per decision: five pre-move fields and five actual-move differences.
  Per game: mean and std across valid sampled decisions. Normal20 features;
  HumanSL60features. No feature selection based on evaluation.
- Reuse original frozen C1+C2 members. Five rank-stratified player folds,
  shared20questions/player; fit fusion and C1 only on fitting players.
- Compare original C1 at lambda0.01, matched C1 at lambda0.1, add normal
  context at lambda0.1, add normal+human context at lambda0.1. Same standard
  shift model and300Adam steps. The regularization control prevents
  attributing a mere regularization change to new signal.
- Primary two contrasts: human+normal versus normal, and versus original
  production. Player-clustered97.5% intervals for the two contrasts.
- Advance only for >=0.005 full-pipeline gain over original, positive lower
  bound for both primary contrasts, no rank loss>0.05. Otherwise reject.
- This historically studied validation cohort is exploratory; even a
  positive screen requires fresh-player confirmation before adoption.

Missing/illegal positions are masked, never silently substituted as valid
zero tokens. Report per-game coverage. Imputation for missing games must
be trained within calibration folds, with missingness explicitly reported.

## Resources

GPU1 only. GPU0 vLLM and other-session files untouched. Raw games, baseline
models and original CSV read-only. Feature shards and evaluation outputs
under production/human_context_20261109; source/tests in this directory.
No automatic commit, push or Aidea submission.
