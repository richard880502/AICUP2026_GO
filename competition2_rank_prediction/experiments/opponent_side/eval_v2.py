"""Paired comparison of the corrected opponent-side pilot (results_v2, 3 seeds each, same 3,000
train / 2,000 validation games).

Per-game metrics (the pilot validation players have too few games for 5-20 game questions): top-1,
competition score of the expected-score decision, NLL; paired player bootstrap for differences vs
player-only. Also late fusion = mean log-probability of the separately trained player and opponent
models (no fitted weights).
Run from the project root: c1/.venv/bin/python experiments/opponent_side/eval_v2.py
"""
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
# eval_common first: rank_lib puts c1/ (which has its own `experiments` package) on sys.path
from experiments import eval_common as ec  # pylint: disable=import-error,wrong-import-position
import rank_lib as lib  # pylint: disable=import-error,wrong-import-position
import transformer_rank as tr  # pylint: disable=import-error,wrong-import-position

E1 = np.exp(-1)
METRICS = ("score", "top1", "nll")
REPORT = HERE / "results_v2/eval_v2.json"


def load(tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validation probabilities, labels, players and game keys of one pilot variant."""
    z = np.load(HERE / f"results_v2/val_probs_opp_{tag}.npz", allow_pickle=True)
    return z["transformer"], z["y"], z["player"], z["game_key"]


def per_game(probs: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Per-game competition score of the decision that maximises expected score, plus top-1 and
    NLL."""
    gain = probs + E1 * (np.pad(probs, ((0, 0), (1, 0)))[:, :-1]
                         + np.pad(probs, ((0, 0), (0, 1)))[:, 1:])
    pred = gain.argmax(1)
    score = np.where(pred == y, 1.0, np.where(np.abs(pred - y) == 1, E1, 0.0))
    return {"score": score, "top1": (probs.argmax(1) == y).astype(float),
            "nll": -np.log(probs[np.arange(len(y)), y] + 1e-9)}


def late_fusion(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Normalised geometric mean of two probability arrays."""
    fused = np.exp((np.log(first + 1e-9) + np.log(second)) / 2)
    return fused / fused.sum(1, keepdims=True)


def summarise(metrics: dict[str, np.ndarray], base: dict[str, np.ndarray] | None,
              boots: list[np.ndarray]) -> dict[str, Any]:
    """Mean metrics, plus paired player-bootstrap differences vs `base` unless base is None."""
    row: dict[str, Any] = {met: float(metrics[met].mean()) for met in METRICS}
    if base is not None:
        for met in ("score", "nll"):
            diff = np.array([(metrics[met][b] - base[met][b]).mean() for b in boots])
            row[f"{met}_minus_player"] = [float(diff.mean()), float(np.percentile(diff, 2.5)),
                                          float(np.percentile(diff, 97.5))]
    return row


def head(name: str, row: dict[str, Any]) -> str:
    """Name and mean metrics of one printed row."""
    return f"{name:55s} score {row['score']:.4f} top1 {row['top1']:.4f} nll {row['nll']:.4f}"


def cross_probs(keys: np.ndarray) -> np.ndarray:
    """Cheap route: the PLAYER-trained models applied to the opponent's decisions (no
    opponent-specific training); normalised geometric mean over the seeds."""
    opp = lib.load_features(HERE / "features/pilot_opp/opponent/val")
    assert (opp["game_key"] == keys).all()
    dev = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    probs = ec.checkpoint_probs(sorted((HERE / "results_v2").glob("opp_player_seed*.pt")),
                                tr.decision_tokens(opp["tokens"]), opp["masks"], dev)
    p_cross = np.exp(np.mean(np.log(np.stack(probs) + 1e-9), 0))
    p_cross /= p_cross.sum(1, keepdims=True)
    return p_cross


def validation_variants() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Validation probabilities of every pilot variant (incl. late fusion), y, player, keys."""
    pp, y, player, keys = load("player")
    variants = {"player only": pp}
    for tag, name in (("opponent", "opponent only"),
                      ("both", "player + opponent (192 decisions)")):
        probs, y2, _, k2 = load(tag)
        assert (k2 == keys).all() and (y2 == y).all()
        variants[name] = probs
    variants["late fusion: mean log-prob of player & opponent models"] = late_fusion(
        variants["player only"], variants["opponent only"] + 1e-9)
    return variants, y, player, keys


def player_boots(player: np.ndarray) -> list[np.ndarray]:
    """2,000 player-bootstrap resamples (seed 0) as game-index arrays."""
    players = np.unique(player)
    idx = {q: np.flatnonzero(player == q) for q in players}
    rng = np.random.default_rng(0)
    return [np.concatenate([idx[q] for q in rng.choice(players, len(players))])
            for _ in range(2000)]


def main() -> None:
    """Writes results_v2/eval_v2.json (first without, then with the cheap-route rows)."""
    variants, y, player, keys = validation_variants()
    metrics = {k: per_game(v, y) for k, v in variants.items()}
    boots = player_boots(player)
    report = {}
    base = metrics["player only"]
    print(f"{len(y)} validation games, {len(np.unique(player))} players")
    for k, met in metrics.items():
        row = summarise(met, None if k == "player only" else base, boots)
        report[k] = row
        extra = "" if k == "player only" else (
            f"   score {row['score_minus_player'][0]:+.4f} [{row['score_minus_player'][1]:+.4f}, "
            f"{row['score_minus_player'][2]:+.4f}]   nll {row['nll_minus_player'][0]:+.4f} "
            f"[{row['nll_minus_player'][1]:+.4f}, {row['nll_minus_player'][2]:+.4f}]")
        print(f"{head(k, row)}{extra}")
    REPORT.write_text(json.dumps(report, indent=2))

    p_cross = cross_probs(keys)
    cheap = {"player model on opponent moves": p_cross,
             "late fusion with player model on opponent moves":
                 late_fusion(variants["player only"], p_cross)}
    for k, probs in cheap.items():
        row = summarise(per_game(probs, y), base, boots)
        report[k] = row
        print(f"{head(k, row)}   score {row['score_minus_player'][0]:+.4f} "
              f"[{row['score_minus_player'][1]:+.4f}, {row['score_minus_player'][2]:+.4f}]   "
              f"nll {row['nll_minus_player'][0]:+.4f}")
    REPORT.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
