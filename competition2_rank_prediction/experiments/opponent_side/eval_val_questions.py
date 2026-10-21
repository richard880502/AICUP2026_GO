"""Supporting check on the validation cohort: the same 8,000 simulated questions as
experiments/strength_axis (20 per player, 5-20 games), 5-fold player-disjoint CV for the fusion
weights and per-rank offsets (fitted on 4 folds, scored on the 5th).
Run from the project root after eval_holdout.py:
    c1/.venv/bin/python experiments/opponent_side/eval_val_questions.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments import eval_common as ec  # pylint: disable=import-error,wrong-import-position

HERE = ROOT / "experiments/opponent_side"
FILES = {"d96_h": (ROOT / "results/val_probs_d96_h.npz", "transformer"),
         "resnet8c_h": (ROOT / "results/val_probs_resnet8c_h.npz", "resnet8c_h"),
         "d96_h_opp": (HERE / "results_v2/val_probs_d96_h_opp.npz", "transformer")}
CONFIGS = {"best2": ["d96_h", "resnet8c_h"],
           "best2 + opponent": ["d96_h", "resnet8c_h", "d96_h_opp"],
           "d96_h": ["d96_h"], "d96_h + opponent": ["d96_h", "d96_h_opp"],
           "opponent alone": ["d96_h_opp"]}
COMPARISONS = (("best2 + opponent", "best2"), ("d96_h + opponent", "d96_h"),
               ("d96_h + opponent", "best2"))


def main() -> None:
    """Scores every configuration and writes results_v2/eval_val_questions.json."""
    logs = {m: ec.clipped_log(np.load(f, allow_pickle=True)[k]) for m, (f, k) in FILES.items()}
    ref = np.load(FILES["d96_h"][0], allow_pickle=True)
    y, player = ref["y"], ref["player"]
    questions = ec.validation_questions(y, player)
    res = {name: ec.cv_question_scores(np.stack([logs[m] for m in models]), y, player, questions)
           for name, models in CONFIGS.items()}
    rep: dict[str, float | list[float]] = {k: float(s.mean()) for k, s in res.items()}
    for k, s in res.items():
        print(f"{k:20s} {s.mean():.4f}")
    rng = np.random.default_rng(1)
    for first, second in COMPARISONS:
        interval = ec.player_bootstrap(res[first] - res[second], questions, rng)
        rep[f"{first} minus {second}"] = interval
        print(f"{first} minus {second}: {ec.format_interval(interval)}")
    (HERE / "results_v2/eval_val_questions.json").write_text(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
