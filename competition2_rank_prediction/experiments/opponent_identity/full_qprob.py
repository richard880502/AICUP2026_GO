"""Replays the cross-fitted C1+C2 pipeline on holdout3 (read-only).

Saves per-question probabilities plus per-game member probabilities to
scratch. Never calls evaluate.main (which writes report.json).

Reproduces Competition 2 rank submission v47, step 2 of 8: writes
`rank_probe/full_qprob.npz`, whose `qfull` (C1+C2 question probabilities)
is the holdout3 base distribution of h3_dump.py.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification python -m \
      competition2_rank_prediction.experiments.opponent_identity.full_qprob
"""
import importlib.abc
import importlib.machinery
import importlib.util
import pathlib
from typing import cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

ROOT = pathlib.Path("runtime/competition2_research").resolve()
# The research machine's location for the rank-probe intermediate files.
OUT = pathlib.Path("/tmp2/richard/claude_scratch/rank_probe")


def main() -> None:  # pylint: disable=too-many-locals
    """Replays both holdout3 halves and writes full_qprob.npz."""
    for j in (0, 1):  # refuse to run if replay would create files in the research workspace
        assert (ROOT / f"results/val_probs_resnet8c_c{j}_bestepoch.npz").exists()
        assert (ROOT / f"experiments/crossfit_cohort/results/probs_c{j}.npz").exists()
    spec = importlib.util.spec_from_file_location(
        "cf_eval", ROOT / "experiments/crossfit_cohort/evaluate.py"
    )
    ev = importlib.util.module_from_spec(cast(importlib.machinery.ModuleSpec, spec))
    cast(importlib.abc.Loader, cast(importlib.machinery.ModuleSpec, spec).loader).exec_module(ev)
    lib, blend_eval, residual_shift = ev.lib, ev.blend_eval, ev.residual_shift
    ev.torch.set_num_threads(8)
    lab = pd.read_csv(ROOT / "features/holdout3/question_folds.csv")
    h3_opp = lib.load_features(ROOT / "features/feat96opp/holdout3")
    fins = (residual_shift.load_final(ROOT / "features/final_result/val"),
            residual_shift.load_final(ROOT / "features/final_result/holdout3"))
    qfull = np.zeros((len(lab), 10))
    sc = np.zeros(len(lab))
    games = {}
    for j in (0, 1):
        half, rows_j = ev.load_half(j, lab, h3_opp, fins)
        members = ["d96", "res", "oppC2"]
        vlogs = np.log(np.stack([half.val[m] for m in members]))
        tlogs = np.log(np.stack([half.test[m] for m in members]))
        w = blend_eval.fit_weights(vlogs, half.vy)
        q = ev.residual_probs(half, "oppC2", (vlogs, tlogs), w)
        qfull[rows_j] = q
        sc[rows_j] = ev.score(q, half.ty)
        d = np.load(ev.member_dump(half.sfx), allow_pickle=True)
        games[j] = {"game_key": d["game_key"].astype(str), "qid": d["player_id"].astype(str),
                    "d96": half.test["d96"], "res": half.test["res"], "oppC2": half.test["oppC2"],
                    "fused": blend_eval.fuse(tlogs, w)}
        print("half", j, "C1+C2", sc[rows_j].mean(), "weights", np.round(w, 3), flush=True)
    print("C1+C2 replay", sc.mean())
    np.savez(OUT / "full_qprob.npz", qfull=qfull, score=sc,
             qid=lab["question_id"].values.astype(str),
             rank=lab["rank"].values.astype(str), fold=lab["fold"].values,
             **{f"{k}_c{j}": v for j, half_games in games.items() for k, v in half_games.items()})


if __name__ == "__main__":
    main()
