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


## P3: Known vs unknown opponent recognition (frozen v4 controls, exploratory)

`known_unknown_audit.py` answers a different question from the rank prior
experiments: **can v4 separate previously seen gallery identities from players
whose identities it never saw?** The 4,096 `present.csv` known-identity games
are compared against the 18,000 games of 1,200 never-seen validation/final
cohort players in the existing `opp_probe/controls.npz`. No encoder is trained.

Run from the repository root (CPU only, using existing caches):

```bash
git fetch origin
git switch experiment/opponent-rank-posterior-audit
git pull --ff-only origin experiment/opponent-rank-posterior-audit
bash competition2_rank_prediction/experiments/opponent_rank_posterior_audit/run_known_unknown_audit.sh
```

The runner requires `/tmp2/richard/claude_scratch/opp_probe/{controls.npz,present.csv}`.
Override `PROBE`, `CONTROLS`, `PRESENT`, `PY`, and `OUT` as needed. If
`h3_sims.pt` (original v4 holdout3 game scores) and the prior
`v47_error_decomposition.npz` exist, a **read-only v47 proxy diagnostic** is
included. It also tries the archived v4 cache at
`production/rank_opponent_identity_20261114/h3_sims.pt`. If either file is
missing, the control result still runs; the v47 proxy is skipped.

Outputs in `$OUT`:

- `known_unknown_audit.json`: held-out known/unknown AUC, player-balanced Brier,
  known recall at 1% / 5% false-positive rate (cutoffs determined from the
  *opposite* controls fold), and illustrative precision at assumed known shares
  of 50% and 70%. Includes optional v47 proxy quartile tables.
- `known_unknown_audit.npz`: control OOF scores/labels/group folds and optional
  question-aligned v47 proxies; no official answers or new submissions.

Implementation checks / assumptions:

1. Original `link_test.absent_controls()` concatenates the 600x15 validation
   cohort and 600x15 final cohort in player-major order, so the 18,000 absent
   scores are grouped into **1,200 disjoint players**, 15 consecutive games
   each. The loader intentionally refuses a different layout rather than
   silently mixing opponents from the same player across folds.
2. Two control folds are assigned by a stable player hash. Model selection
   uses a **fixed 2-feature (top cosine, top-two margin), ridge-regularized
   logistic regression** trained on opposite-fold controls. Each class and
   each player have equal weight within the fitting fold. The score has a
   balanced-control prior; it is not deployment-calibrated `P(known)`.
3. Optional v47 features come from **v4** `h3_sims.pt` and align 1:1
   with frozen `v47_error_decomposition.npz` question IDs, own-rank truth,
   folds and HD0/HD1 eligibility. No holdout3 labels are used in fitting
   the classifier; labels are used only to *describe* OOF utility/error rates.
4. **Do not deploy the transferred classifier:** v47 masks  holdout3
   player classes and uses v4+v7 similarity averaging. The v4-only controls
   use the full unmasked gallery, so control calibration does not transfer
   faithfully to v47. The optional question score is explicitly named
   `v4_known_proxy`, not `P(true opponent is known)`. The masked-gallery
   distribution shift is recorded. There are no true opponent identity labels
   on holdout3, so this is *not* an opponent oracle or causal attribution.
5. No v47 model weights, class priors, decoder or Production code are changed.
   The report always sets `production_gate_passed=false`; all correlations,
   error strata and within-confidence comparisons remain exploratory because
   Holdout3 was repeatedly inspected.

Synthetic CPU tests:

```bash
python -m unittest competition2_rank_prediction.experiments.opponent_rank_posterior_audit.test_known_unknown_audit -v
```

**Research decision:** Good control AUC alone cannot justify a new v47 gate
or deployment. The next *properly controlled* experiment would reproduce
the controls using the **same gallery mask and v4+v7 ensemble** as v47, and
then test any tiny gating head on a fresh nested player-disjoint outer cohort.
