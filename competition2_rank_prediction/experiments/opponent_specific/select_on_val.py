"""Choose the opponent-specific Transformer variant on the VALIDATION cohort only (holdout3 is not
touched).

Members: d96_h2 + resnet8c_h2 + an opponent member. The borrowed member (d96_h2 on opponent moves =
current default) is compared with the three opponent-specific variants trained by run_variants.sh.
Metrics: per-game NLL of the opponent member, and the question-level score of the full fusion on the
strength_axis questions (20 per player) with 5-fold player-disjoint fitting of weights and offsets.
Run from the project root: c1/.venv/bin/python experiments/opponent_specific/select_on_val.py
"""
import json
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import eval_common as ec  # pylint: disable=import-error,wrong-import-position

HERE = ROOT / "experiments/opponent_specific"
BASE = {"d96_h2": (ROOT / "results/val_probs_d96_h2.npz", "transformer"),
        "resnet8c_h2": (ROOT / "results/val_probs_resnet8c_h2.npz", "resnet8c_h2")}
REF_NAME = "borrowed d96_h2 (current default)"
OPP = {REF_NAME: ROOT / "results/val_probs_d96_h2_opp.npz",
       **{f"opponent-specific {v}": HERE / f"results/val_probs_opp_{v}_h2.npz"
          for v in ("head", "full", "scratch")}}


def aligned(path: pathlib.Path, keys: np.ndarray, key: str = "transformer") -> np.ndarray:
    """Clipped log-probabilities of `path`[key], rows in the order of `keys`."""
    z = np.load(path, allow_pickle=True)
    index = {k: i for i, k in enumerate(z["game_key"])}
    return ec.clipped_log(z[key][[index[k] for k in keys]])


def main() -> None:
    """Scores every opponent member and writes results/select_on_val.json."""
    ref = np.load(BASE["d96_h2"][0], allow_pickle=True)
    keys, y, player = ref["game_key"], ref["y"], ref["player"]
    base_logs = [aligned(f, keys, k) for f, k in BASE.values()]
    questions = ec.validation_questions(y, player)

    res: dict[str, np.ndarray] = {}
    report: dict[str, dict[str, Any]] = {}
    for name, path in OPP.items():
        if not path.exists():
            print("missing", path)
            continue
        opp_logs = aligned(path, keys)
        res[name] = ec.cv_question_scores(np.stack(base_logs + [opp_logs]), y, player, questions)
        report[name] = {"opp_member_nll": float(-opp_logs[np.arange(len(y)), y].mean()),
                        "question_score": float(res[name].mean())}
    rng = np.random.default_rng(1)
    for name, scores in res.items():
        entry = report[name]
        entry["minus_default"] = ec.player_bootstrap(scores - res[REF_NAME], questions, rng)
        print(f"{name:40s} member NLL {entry['opp_member_nll']:.4f}   "
              f"question score {entry['question_score']:.4f}   "
              f"vs default {entry['minus_default'][0]:+.4f} [{entry['minus_default'][1]:+.4f}, "
              f"{entry['minus_default'][2]:+.4f}]")
    (HERE / "results/select_on_val.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
