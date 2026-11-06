# AICUP 2026 Go — Competition 2 Rank Prediction

This repository records the reproducible code, validation protocol, and
experiment conclusions for predicting a player's rank from 5–20 anonymized
19x19 Go games. Licensed SGF data, generated features, KataGo/HumanSL weights,
and trained checkpoints are intentionally excluded from version control.

Start with:

- [`CURRENT_BEST.md`](CURRENT_BEST.md): authoritative current pipeline and
  locked holdout results.
- [`METHODS.md`](METHODS.md): detailed Traditional-Chinese research record,
  including rejected experiments and pitfalls.
- [`INDEX.md`](INDEX.md): source-code, artifact, and test-day map.

The user-reported official Public score for the 2026-10-30 test submission
is **0.7227214**. The complete record is in
[`docs/public_submission_20261030.md`](docs/public_submission_20261030.md).

## Update 2026-10-21: opponent-side evidence (current default)

The labelled player's **opponents** turned out to be strong evidence of the player's *registered* rank: games are matched by
the server rank, so averaging the opponents over 5-20 games tracks the label, while the player's own moves keep a persistent
player-specific offset. (A 2026-10-12 experiment had rejected this idea, but `extract_rank_features.py --opponent` ignored its
flag and re-extracted the player's own moves; fixed.)

| Step (all internal, player-disjoint) | Result |
| --- | --- |
| best2 + dense Transformer applied to the opponent's moves | +0.053 validation, +0.020 old holdout, +0.067 fresh 400-player cohort (0.6535 -> 0.7206) |
| Pre-registered 2,000-player cross-fitted test: C1 question-level shift residual (opponent distribution, player x opponent pairing, KataGo final-position result proxy) | +0.030 [+0.017, +0.043], passes Holm |
| same test: C2 Transformer trained to predict the player's rank from the opponent's moves | +0.007 [+0.001, +0.013], passes Holm |
| C1 + C2 (now the `predict_test.py` default) | 0.6763 -> 0.7170 (+0.041 [+0.027, +0.054]) |

Details: `experiments/opponent_side`, `experiments/fresh_cohort`, `experiments/crossfit_cohort` (with `PREREGISTRATION.md`),
`experiments/a1d_boundary` (why A and 1D overlap: their playing strength differs by only ~0.2-0.3 rank units, also through the
opponents). Test day: `scripts/run_test_extraction.sh` (player + opponent passes + final-position features), then
`predict_test.py`. The sections below describe the state before this update.

## Current result (2026-10-13)

All numbers below are internal player-disjoint evaluations, not official
leaderboard scores.

| Method | 400-player holdout score |
| --- | ---: |
| Official tutorial reproduction | 0.4031 |
| Stable two-model production | 0.6609 |
| Dense player-bag + rank logits | 0.6723 |
| **Cross-feature dense player-bag candidate** | **0.6780** |

The current candidate combines:

```text
SGF
  -> up to 96 player decision points
  -> KataGo normal policy + HumanSL 5k / 1d / 5d
  -> 3-seed ordinal Transformer + cosine ResNet snapshots
  -> per-game rank logits and frozen embeddings
  -> 414D and 606D cross-game Player-Bag Transformers
  -> validation-locked geometric fusion
  -> expected competition-utility decoding
```

Its exact locked score is `0.677955`. The paired confidence intervals against
the previous candidates cross zero, so it remains a candidate until confirmed
on a fresh player-disjoint outer cohort.

## Repository scope

Tracked:

- feature-extraction, training, inference, and evaluation source;
- shell launchers and configuration templates;
- small JSON reports and experiment documentation;
- accepted and rejected experiment records.

Not tracked:

- competition CSV/SGF data;
- `.pt`, `.npz`, `.npy`, `.pkl`, and KataGo/HumanSL model files;
- generated feature shards, embeddings, logs, and virtual environments.

## Environment

Python 3.11 and CUDA-capable PyTorch were used. Create the environment inside
the repository, install a PyTorch build matching the machine's CUDA version,
then install the remaining dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
# Install torch for the host CUDA version first.
.venv/bin/pip install -r requirements.txt
git config core.hooksPath .githooks
```

The local pre-commit hook runs `pylint` and `mypy` on staged Python files.

## Stable production path

Feature extraction and model training require the licensed data plus the
KataGo/HumanSL assets described in `METHODS.md`:

```bash
scripts/run_extraction96.sh

.venv/bin/python transformer_rank.py \
  --train features/feat96/train \
  --val features/feat96/val

.venv/bin/python resnet_rank.py --epochs 5 --cosine --tag resnet5_h
```

Reproduce the validation-selected production comparison:

```bash
.venv/bin/python -m experiments.model_selection.evaluate
```

Reproduce the locked cross-feature candidate when its excluded artifacts are
available locally:

```bash
.venv/bin/python -m \
  experiments.dense_player_bags.evaluate_cross_feature_holdout \
  --device cpu
```

## Important conclusion

Increasing model size, global class-count decoding, richer aggregate
statistics, and a full KataGo value/ownership branch did not transfer reliably
to the locked holdout. The next useful direction is better ability-specific
feature extraction: HumanSL rank-likelihood slopes, position-difficulty
normalized regret, critical-error sampling, and post-error recovery behavior.
