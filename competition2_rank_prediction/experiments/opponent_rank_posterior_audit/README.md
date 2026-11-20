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

## P2 diagnostic: v47 error-source decomposition (exploratory; no new encoder)

**Purpose:** In which circumstances does game-specific opponent rank retrieval
*actually* help v47, and how sensitive are the remaining errors to the
retrieval posterior versus the learned opponent-to-own-rank transition?

**Important:** `h3_sims_v47.pt` has the asked player's true rank, not the
opponents' true identities/ranks. A true opponent-rank Oracle and causal
retrieval-vs-transition error percentages **cannot be computed**. The script
reports only observed prediction flips and counterfactual sensitivities. It
always sets `production_gate_passed=false` and never writes an official answer.

From the repository root, on the machine with the existing v47 cache:

```bash
git fetch origin
git checkout experiment/opponent-rank-posterior-audit
GPU=1 bash competition2_rank_prediction/experiments/opponent_rank_posterior_audit/run_decomposition.sh
```

No KataGo extraction or frozen encoder training. The script refits two tiny
v47 cross-fit models per fold for a **properly refitted information ablation**
(full versus no game-specific identity), and applies four fixed *same-model*
perturbations (posterior temperature x0.75/x1.25, transition-logit scale
x0.75/x1.25). The full arm **must replay 0.765302** on the original 2,000
holdout3 questions before a report is written. No new model is adopted.

Requires these existing inputs:

- `/tmp2/richard/claude_scratch/opp_probe/h3_sims_v47.pt`;
- `runtime/competition2_research/features/holdout3/question_folds.csv`.

Override `PY=`, `GPU=`, `DEVICE=`, `STATE=`, `LABELS=` and `OUT=` as necessary.
The script works from any working directory. For CPU-only debugging set
`DEVICE=cpu`, but full 300-step cross-fitting on ~6,807 identity classes will
be much slower than GPU.

Outputs in the same `OUT` folder as P0:

- `v47_error_decomposition.json`: replay checks, `full` / `no_retrieval` /
  own-move `base` utility and NLL, paired player-bootstrap CIs, 1D/2D and
  other hard-boundary AUC, confidence bins, and four-way sensitivity table.
- `v47_error_decomposition.npz`: **aligned per-question** OOF predictions,
  true own rank, question/player IDs, folds, confidence proxies, error/sensitivity
  masks, eligibility and number of usable games for audit/reanalysis.

Interpretation:

1. `retrieval_increment_full_minus_refitted_no_retrieval` estimates the value
   of **game-specific opponent retrieval evidence** after refitting, not the
   proportion of wrong opponent identities. The no-retrieval arm still knows
   game kind, game count and C1+C2 base.
2. `base_full_error_table` counts rescue/hurt/both-wrong when adding the **whole
   identity+transition stack**. It does not isolate an opponent retrieval error.
3. `error_sensitivity_not_causal_attribution` groups wrong questions by whether
   ±25% perturbations to *frozen* retrieval/transition parameters flip the
   chosen rank. These perturbations are deliberately not matched in causal
   intensity. `neither` means stable to **these perturbations**, not inherently
   unfixable or registered-rank label noise.
4. `official_compatible_questions` excludes questions with any HD>=2 game.
   Report all-question and filtered results, not just the best-looking subset.
5. Holdout3 is repeatedly inspected and v47 weights are fit cross-fold, so this
   remains **exploratory**. Do not use it as an independent production gate or
   decide to train a new Rank-aware Head solely from the diagnostic.

CPU tests (synthetic, no competition data):

```bash
python -m unittest competition2_rank_prediction.experiments.opponent_rank_posterior_audit.test_decompose_v47 -v
```
