# Release v1.1_2026-10-22: frozen official pipeline (current)

Frozen on 2026-10-22 at about 15:30 Taipei time; reference outputs were finished at about 17:05.

**v1.1 = v1 + input hardening.**
- The models, model weights, validation inputs and fitted parameters are unchanged from
  `release/v1_2026-10-22`, which stays as the rollback.
- The method is unchanged: best2 + C2 opponent member (opp_scratch) + C1 shift residual + expected-score
  decoding.
- Pre-registered estimate on 2,000 unseen players (experiments/crossfit_cohort): **0.7170**.

## What changed from v1

| File | Change |
|---|---|
| `sgf_check.py` (new) | SGF sanitizer: setup stones become leading moves, `tt` passes, whitespace / BOM / lower-case properties, comments, off-board or truncated records, occupied points, suicide and immediate ko recapture are cut, non-19x19 games are dropped. It changes 0 of 95,804 real games |
| `scripts/test_to_games.py` | Uses the sanitizer and every non-empty `sgf_i`; normalises or drops bad colours; keeps the first of duplicate ids; writes `input_report.csv` |
| `scripts/prep_test_moves.py` | Same sanitizer; ids read as text |
| `predict_test.py` | Per-question fallback chain (default → 3-member + offsets → player+opponent → ResNet+opponent → best2 → opponent only → player only → ResNet only → prior 1D). Each question takes the first method that can use at least half of its usable games. Also: `<out>_report.csv`, `--report`, `--strict`; NaN or missing features are excluded instead of being decoded as D; numeric-looking ids are read as text; a wrong question count is reported after writing the submission. `--model-set full4` keeps the old path |
| `residual_shift.py` | Only adds a `FINAL_NAMES` constant |
| `extract_rank_features.py` | The KataGo client keeps draining after a per-query error, so it no longer deadlocks: a rejected game is masked and the shard completes. A ko-illegal decision is masked instead of raising. Tolerant SGF parsing. Opt-in `--deterministic` |
| `scripts/extract_final_result.py` | Rejected games get NaN values, which `predict_test.py` treats as missing. Tolerant parser; opt-in `--deterministic` |
| `scripts/run_test_extraction.sh` | `ATTEMPT_TIMEOUT` (default 5400 s per attempt, retried, finished shards kept); output overrides (`SHARDS`, `PLAYER_OUT`, `OPP_OUT`, `FINAL_OUT`, `LOG_DIR`, `GPUS`); `DETERMINISTIC` |
| `scripts/e2e_check.py`, `preflight_test.py`, `check_submission.py`, `verify_release.py`, `run_frozen.sh` (new) | Golden end-to-end check, release-day preflight, submission check, release tooling. `run_frozen.sh` and `verify_release.py` default to v1.1; `RELEASE=` / `--release` selects v1 |
| `tests/anomaly/run_anomaly.py` (new) | Malformed-input suite. All checks pass, including the real-KataGo stage |

Details of the hardening are in `release/golden/VERIFY.md` and the 2026-10-22 ANOMALY / VERIFY reports.

## Layout
This is the same as v1; see `release/v1_2026-10-22/README.md`.

| Path | Contents |
|---|---|
| `code/` + `code_sha256.txt` | Read-only snapshot of the live code at freeze time (48 files, including `tests/anomaly/`) |
| `runtime/` | The frozen project root: real copies of `code/`, plus links `c1`, `results` and `features/{feat96opp,final_result}/val` → `artifacts/` |
| `artifacts/` | Hard links to v1's artifacts: the same inodes, read-only and hashed in this manifest, with no second 1 GB copy |
| `fitted/` | Re-dumped from the v1.1 runtime (`OMP_NUM_THREADS=8`) |
| `reference_outputs/` | See below |
| `MANIFEST.json`, `ENV.txt`, `tools/` | As in v1. The environment was re-checked: pip freeze, driver and kernel are identical to v1 |

## Identity with v1
- **Fitted parameters:** all 20 arrays (fusion weights, C1 mu/sd/b/w, offsets for default, opp_d96, no_residual,
  best2 and full4) are `np.array_equal` to v1's `fitted/fitted_params.npz`. `fitted_params.json` is byte-identical.
- **Code paths on clean data, same device (CPU, holdout2 400 questions, `identity_v1_cpu/` vs `holdout2_cpu/`):**
  - For default, `--no-residual` and `--no-opp`, v1 and v1.1 give byte-identical submissions.
  - Every per-game member probability (d96, resnet8c, opp_scratch) is bit-identical.
  - The 2026-10-22 ANOMALY branch also showed live code (now this snapshot) = v1 on cuda:1 for the same three
    configurations.
- **Against v1's cuda:1 reference:**
  - v1.1 on CPU reproduces `--no-residual`, `--opp-model d96`, `--no-opp` and full4 byte for byte.
  - The default differs on 1 of 400 questions (0.7315 vs 0.7331). This is a device difference, not a code one:
    v1 on CPU gives the same single difference. ResNet probabilities differ by up to 0.0087 between CPU and GPU.
  - **Official runs must use a GPU.**
- **No fallbacks on clean data:** every configuration used its intended method for 400 of 400 questions
  (`holdout2_cpu/pred_*_report.csv`).
  - The opp_d96 report labels its method "default (C1+C2)" even though the opponent member is d96. This is a
    label only.

## Reference outputs
| Path | Contents |
|---|---|
| `holdout2/` | GPU reference: v1's cuda:1 outputs, hard-linked. They are valid for v1.1 by the identity above. Both GPUs were full when v1.1 was frozen (other users' jobs and experiments/extraction_tta) |
| `holdout2_cpu/` | v1.1's own runs of all five configurations on CPU, with per-question fallback reports |
| `identity_v1_cpu/` | v1 on CPU for default, no_residual and best2; the identity evidence |
| `summary.json` | Scores, hashes and the identity flags |

| Configuration | holdout2 GPU | holdout2 CPU (v1.1) |
|---|---|---|
| default | 0.7331 | 0.7315 |
| `--no-residual` | 0.7430 | 0.7430 |
| `--opp-model d96` | 0.7435 | 0.7435 |
| best2 `--no-opp` | 0.6997 | 0.6997 |
| full4 | 0.6884 | 0.6884 |

These numbers are a regression reference only: the production models saw these players. The clean estimate is
0.7170.

## Run, verify, roll back
```bash
cd /tmp2/richard/rank_prediction
scripts/run_frozen.sh check                       # v1.1: manifest + runtime + external hashes, fitted-parameter recompute
PER_GPU=1 scripts/run_frozen.sh all --tests-dir /path/to/tests --device cuda:1 --out /abs/path/submission.csv
c1/.venv/bin/python scripts/verify_release.py     # v1.1 PASS/FAIL + live differences (INFO)
RELEASE=release/v1_2026-10-22 scripts/run_frozen.sh check     # rollback release still passes
```
- The step-by-step test-day procedure is in `release/RUNBOOK.md`.
- Rollback and fallbacks are in [ROLLBACK.md](ROLLBACK.md).

## Open items
- **`DETERMINISTIC=1` (fixed KataGo board symmetry): not for the submission** (experiments/extraction_tta, 2026-10-22: −0.007 [−0.018, +0.003] vs random symmetry; averaging two random passes +0.0004, rejected). Use it only for reproducibility checks.
  - Training and validation features used random symmetry.
  - Fixed symmetry makes reruns identical (golden det1 / det2 / det3), but it is not yet known whether it scores
    the same.
  - Leave it unset until then.
- **512/512 analysis config:** treated as equivalent to the validated 160/256 at KataGo's own run-to-run noise level
  (golden test). The 160/256 template is still frozen; see ROLLBACK §5.
- **cuda:0 vs cuda:1 bit-identity:** not checked. The GPU reference was produced on cuda:1.
