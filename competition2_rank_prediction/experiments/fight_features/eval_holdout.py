"""Untouched 400-player holdout: production best2 (d96_h + cosine ResNet) vs the same with the
fight-feature Transformer.

Fusion weights and per-rank offsets are fitted on the validation cohort only; pooling = log-mean;
expected-score decode.
Run from the project root:  c1/.venv/bin/python experiments/fight_features/eval_holdout.py
"""
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# eval_common first: rank_lib puts c1/ (which has its own `experiments` package) on sys.path
from experiments import eval_common as ec  # pylint: disable=import-error,wrong-import-position
import rank_lib as lib  # pylint: disable=import-error,wrong-import-position
import transformer_rank as tr  # pylint: disable=import-error,wrong-import-position

EXP = ROOT / "experiments/fight_features/results"
VAL_FILES = {"fight_h": (EXP / "val_probs_fight_h.npz", "transformer"),
             "d96_h": (ROOT / "results/val_probs_d96_h.npz", "transformer"),
             "resnet8c_h": (ROOT / "results/val_probs_resnet8c_h.npz", "resnet8c_h")}
CONFIGS = (("production best2: d96_h + resnet8c_h", ["d96_h", "resnet8c_h"]),
           ("fight_h + resnet8c_h", ["fight_h", "resnet8c_h"]),
           ("d96_h alone", ["d96_h"]),
           ("fight_h alone", ["fight_h"]))
COMPARISONS = (("fight_h + resnet8c_h", "production best2: d96_h + resnet8c_h"),
               ("fight_h alone", "d96_h alone"))


def fight_member(feats: dict[str, np.ndarray], device: torch.device) -> np.ndarray:
    """Mean probabilities of the 3 fight_h Transformer seeds on the holdout games."""
    x, valid = tr.decision_tokens(feats["tokens"]), feats["masks"]
    x, valid = tr.add_value_features(x, valid, feats, ROOT / "features/fight96/holdout", 1.0)
    probs = ec.checkpoint_probs(sorted(EXP.glob("fight_h_seed*.pt")), x, valid, device)
    assert len(probs) == 3, f"expected 3 seeds, found {len(probs)}"
    return np.mean(probs, 0)


def main() -> None:
    """Scores every configuration on the holdout and prints paired bootstrap differences."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feats = lib.load_features(ROOT / "features/feat96/holdout")
    # production members on the same games
    members = {"fight_h": fight_member(feats, device),
               **ec.production_members(feats["game_key"])}
    holdout = ec.load_holdout(feats["player_id"])

    results = {}
    for name, models in CONFIGS:
        scores, weights = ec.holdout_scores(models, VAL_FILES, members, holdout)
        results[name] = scores
        print(f"{name:40s} {scores.mean():.4f} +-{scores.std(ddof=1) / np.sqrt(len(scores)):.3f}"
              f"   weights {dict(zip(models, np.round(weights, 3)))}")
    rng = np.random.default_rng(1)
    for first, second in COMPARISONS:
        diff = results[first] - results[second]
        print(f"[{first}] minus [{second}]: "
              f"{ec.format_interval(ec.bootstrap_interval(diff, diff, rng))}")


if __name__ == "__main__":
    main()
