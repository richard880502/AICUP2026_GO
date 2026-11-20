# Opponent Rank Posterior: Ground-truth control audit (P0)

Branch: `experiment/opponent-rank-posterior-audit`. **No new model is trained.**

## Question

v47 obtains evidence from identified opponents' **registered ranks**, not merely their
exact identities. When v4 predicts the wrong identity, how often does the wrong
identity still have the correct **rank**? Does soft rank pooling already recover
the correct rank where hard Top-1 identity fails? Is the remaining retrieval
error particularly severe on A/1D, 1D/2D and 2D/3D?

The sample consists of `present.csv`: 4,096 unused games from **known** v4
training-gallery players. Class identity and rank ground truth are available
from the original training metadata in `note="<class>:<global_row>"`. The games
were excluded from the v4 gallery's training features, although the **players
are the same identities** the encoder was trained to recognize. This controls
unseen-game generalization for *known* identities, NOT unknown-player retrieval.

This is **not** a v47 Oracle score. The official rank test does not include
opponents' true identities; the known-player control cannot establish how much
question-level Utility can be recovered from exact opponent labels. The `oracle
rank = 100%` is true by construction and is only a reference to reveal rank
retrieval headroom.

## Steps (from repository root on the research host)

```bash
git fetch origin
git checkout experiment/opponent-rank-posterior-audit
GPU=1 bash competition2_rank_prediction/experiments/opponent_rank_posterior_audit/run.sh
```

`run.sh` reuses existing `present.csv` and trunk shards where present; creates
`present.csv` from **unused known-player training games only** if absent, then
resumes the same frozen HumanSL local-only extraction (160 decisions). If all
shards already exist, the extractor skips them. It then loads frozen
`style_v4/checkpoint.pt`, computes class similarities, checks class/rank and
record-id alignment, and writes the report. No official test records are read.

Outputs (override `OUT` to choose directory):

```
/mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit/
  present_rank_scores.npz
  report.json
```

Dependencies are the repository's existing runtime environment (PyTorch,
NumPy, SciPy, HumanSL tokenizer / KataGo vendor data). Overrides:
`PY=...`, `GPU=1`, `WORKERS=4`, `BATCH_GAMES=4`, `PROBE=...`, `OUT=...`.
If the host's Python lives in `runtime/competition2_research/c1/.venv/bin/python`,
the script detects it; otherwise set `PY` explicitly. The scorer uses
`train.encode` with the original v4 layout. Importing `rich_full` before v4
is deliberately avoided because of the documented import-time monkeypatch.

### Analyze an existing scored cache only (no GPU)

```bash
PYTHONPATH=. python -m competition2_rank_prediction.experiments.opponent_rank_posterior_audit.audit \
  --scores /mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit/present_rank_scores.npz \
  --output /mnt/share_data_77/richard/AICUP2026_GO/opponent_rank_posterior_audit/report.json
```

### Unit tests (CPU; no competition data)

```bash
python -m unittest competition2_rank_prediction.experiments.opponent_rank_posterior_audit.test_audit -v
```

## Comparisons

1. **Identity Top-1:** does the highest-scoring gallery class match ground truth?
2. **Hard-rank Top-1:** is that class's registered rank correct, even when the
   identity is wrong? Report the fraction of identity errors preserving rank.
3. **Soft-rank posterior:** `softmax(class_cosines / 0.04) @ one_hot(class_rank)`.
   This matches the *rank aggregation* in v47; the temperature `0.04` is its
   initialization, **not** necessarily the temperature fitted by v47.
4. **Cross-fitted temperature:** fixed candidate grid `[0.02, 0.03, 0.04,
   0.06, 0.08, 0.12]`, selected by rank NLL on the opposite **player-disjoint**
   fold, to see whether simple calibration is missing rank mass.
5. **Oracle-rank:** correct class means known rank, so exactly 100% rank
   accuracy. It is a *tautological upper reference*, never an end-to-end gain.

Report includes multiclass rank NLL, Brier, Top-1, adjacent boundary AUCs,
confusions, per-rank player/game support, player-bootstrap 95% paired CIs,
and low/high cosine-margin bins whose cutoffs come from the other fold **without
labels**. Different games from one player never cross the calibration split.

## Pre-declared interpretation

* If identity Top-1 is poor but **hard and soft rank retrieval are already
  strong**, another player-identification encoder may mostly help Comp1,
  not Comp2.
* If the same-rank share of identity mistakes is low and calibrated rank
  NLL remains high, rank-aware representation/calibration is worth a controlled
  follow-up. **Do not infer final v47 utility gain from this alone.**
* If A/1D, 1D/2D, 2D/3D have too few distinct players in the control,
  do not treat their boundary AUC as a strong result.
* To decide whether errors in *registered-rank prediction* are irreducible,
  one needs verified true opponent labels, independent ratings/promotion
  histories, or a separately justified experimental proxy. These do not
  exist in the currently published official-test inputs.

Do not tune representation weights or run another experimental leaderboard
submission based on this P0 report. If there is convincing rank-level
headroom, design a properly player-disjoint **P1 frozen Rank-aware Head** on
training-only games and then a fresh independent **P2** full-pipeline gate.
