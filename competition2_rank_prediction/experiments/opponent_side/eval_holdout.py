"""Untouched 400-player holdout: production best2 (d96_h + cosine ResNet) vs best2 + the opponent's
side of every game.

The opponent member reuses the production d96_h Transformers unchanged, applied to the OPPONENT's
96 decisions of each game (features/feat96opp, extracted with the fixed `--opponent`). Its
validation probabilities are computed the same way on the validation cohort, so fusion weights and
per-rank offsets are fitted on validation only; pooling = log-mean; expected-score decode.
Run from the project root after run_full_opp.sh:
    c1/.venv/bin/python experiments/opponent_side/eval_holdout.py
"""
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# eval_common first: rank_lib puts c1/ (which has its own `experiments` package) on sys.path
from experiments import eval_common as ec  # pylint: disable=import-error,wrong-import-position
import rank_lib as lib  # pylint: disable=import-error,wrong-import-position
import transformer_rank as tr  # pylint: disable=import-error,wrong-import-position

HERE = ROOT / "experiments/opponent_side"
OPP = HERE / "features/feat96opp"
CHECKPOINTS = sorted((ROOT / "results").glob("d96_h_seed*.pt"))
CONFIGS = {"production best2: d96_h + resnet8c_h": (["d96_h", "resnet8c_h"], None),
           "best2 + opponent (weights fitted on validation)":
               (["d96_h", "resnet8c_h", "d96_h_opp"], None),
           "d96_h alone": (["d96_h"], None),
           "d96_h + opponent (weights fitted on validation)": (["d96_h", "d96_h_opp"], None),
           "opponent alone": (["d96_h_opp"], None)}
COMPARISONS = (("best2 + opponent (weights fitted on validation)",
                "production best2: d96_h + resnet8c_h"),
               ("d96_h + opponent (weights fitted on validation)", "d96_h alone"))


def d96_on(feats: dict[str, np.ndarray], keys: np.ndarray, device: torch.device) -> np.ndarray:
    """Production d96_h (3 seeds, prob-mean like predict_test.py) on `feats`, returned in the order
    of `keys`. Games with no opponent decision get a uniform distribution (neutral after
    log-fusion)."""
    index = {k: i for i, k in enumerate(feats["game_key"])}
    missing = [k for k in keys if k not in index]
    assert not missing, f"{len(missing)} games lack opponent features, e.g. {missing[:3]}"
    rows = np.array([index[k] for k in keys])
    x, valid = tr.decision_tokens(feats["tokens"][rows]), feats["masks"][rows]
    safe = valid.copy()
    safe[~valid.any(1), 0] = True          # avoid an all-masked softmax; overwritten below
    probs = np.mean(ec.checkpoint_probs(CHECKPOINTS, x, valid, device, predict_valid=safe), 0)
    probs[~valid.any(1)] = 0.1
    print(f"  opponent member: {len(keys)} games, {(~valid.any(1)).sum()} "
          "without opponent decisions")
    return probs


def validation_files(device: torch.device) -> dict[str, tuple[pathlib.Path, str]]:
    """Validation probability files of the members; the opponent member is computed (and cached)
    in the order of results/val_probs_d96_h.npz."""
    ref = np.load(ROOT / "results/val_probs_d96_h.npz", allow_pickle=True)
    vfile = HERE / "results_v2/val_probs_d96_h_opp.npz"
    if not vfile.exists():
        np.savez(vfile, y=ref["y"], player=ref["player"], game_key=ref["game_key"],
                 transformer=d96_on(lib.load_features(OPP / "val"), ref["game_key"], device))
    val_files = {"d96_h": (ROOT / "results/val_probs_d96_h.npz", "transformer"),
                 "resnet8c_h": (ROOT / "results/val_probs_resnet8c_h.npz", "resnet8c_h"),
                 "d96_h_opp": (vfile, "transformer")}
    for path, _ in val_files.values():
        assert (np.load(path, allow_pickle=True)["game_key"] == ref["game_key"]).all(), path
    return val_files


def holdout_members(device: torch.device) -> tuple[dict[str, np.ndarray], ec.Holdout]:
    """Holdout member probabilities on the production game order, and the holdout questions."""
    feats = lib.load_features(ROOT / "features/feat96/holdout")
    members = ec.production_members(feats["game_key"])
    members["d96_h_opp"] = d96_on(lib.load_features(OPP / "holdout"), feats["game_key"], device)
    return members, ec.load_holdout(feats["player_id"])


def score_configs(val_files: dict[str, tuple[pathlib.Path, str]], members: dict[str, np.ndarray],
                  holdout: ec.Holdout) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Per-question holdout scores of every configuration, and their report entries."""
    results: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {}
    print("holdout questions:", len(holdout.question_ids))
    for name, (models, fixed_w) in CONFIGS.items():
        scores, weights = ec.holdout_scores(models, val_files, members, holdout, fixed_w)
        shown = dict(zip(models, np.round(weights, 3).tolist()))
        results[name] = scores
        report[name] = {"score": float(scores.mean()),
                        "se": float(scores.std(ddof=1) / np.sqrt(len(scores))), "weights": shown}
        print(f"{name:50s} {scores.mean():.4f} +-{report[name]['se']:.3f}   weights {shown}")
    return results, report


def main() -> None:
    """Scores every configuration on the holdout and writes results_v2/eval_holdout.json."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert len(CHECKPOINTS) == 3
    val_files = validation_files(device)
    members, holdout = holdout_members(device)
    results, report = score_configs(val_files, members, holdout)
    rng = np.random.default_rng(1)
    for first, second in COMPARISONS:
        diff = results[first] - results[second]
        interval = ec.bootstrap_interval(diff, diff, rng)
        report[f"[{first}] minus [{second}]"] = interval
        print(f"[{first}] minus [{second}]: {ec.format_interval(interval)}")
    (HERE / "results_v2/eval_holdout.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
