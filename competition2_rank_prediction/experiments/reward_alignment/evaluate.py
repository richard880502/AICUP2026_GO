# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Evaluates reward-alignment replacements inside the original C1+C2 pipeline.

Example: PYTHONPATH=competition2_rank_prediction python -m
competition2_rank_prediction.experiments.reward_alignment.evaluate
"""

import argparse
import json

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    calibration,
)
from competition2_rank_prediction.experiments.reward_alignment import train
from competition2_rank_prediction.experiments.semi_hard_mining import statistics

ARMS = {
    "nll_nll": ("nll", "nll"),
    "nll_utility": ("nll", "utility"),
    "reward_utility": ("reward", "utility"),
}
MEMBERS = ("d96", "resnet8c", "opp_scratch")


def load_data(args: argparse.Namespace) -> dict[str, np.ndarray]:
    """Loads original validation predictions aligned by immutable game keys."""
    data = {}
    for index, member in enumerate(MEMBERS):
        with np.load(
            args.research / f"results/val_probs_{member}.npz",
            allow_pickle=True,
        ) as payload:
            if index == 0:
                data = {
                    key: np.asarray(payload[key])
                    for key in ("y", "player", "game_key")
                }
            lookup = {
                str(key): row for row, key in enumerate(payload["game_key"])
            }
            rows = np.asarray([lookup[str(key)] for key in data["game_key"]])
            field = "resnet8c" if member == "resnet8c" else "transformer"
            data[member] = np.asarray(payload[field])[rows]
            if not np.array_equal(np.asarray(payload["y"])[rows], data["y"]):
                raise ValueError(f"Mismatched validation labels: {member}")
    final = residual_shift.load_final(
        args.research / "features/final_result/val"
    )
    data["final"] = np.stack([final[str(key)] for key in data["game_key"]])
    for arm, (loss, selection) in ARMS.items():
        values = []
        for seed in range(args.seeds):
            with np.load(
                args.output / f"seed_{seed}" / loss / f"{selection}_outer.npz"
            ) as payload:
                lookup = {
                    str(key): row for row, key in enumerate(payload["game_key"])
                }
                rows = np.asarray(
                    [lookup[str(key)] for key in data["game_key"]]
                )
                values.append(np.asarray(payload["probabilities"])[rows])
        data[arm] = np.stack(values)
    return data


def game_folds(data: dict[str, np.ndarray]) -> np.ndarray:
    """Assigns rank-balanced player-disjoint folds before fitting calibration."""
    generator = np.random.default_rng(20261108)
    folds = np.full(len(data["y"]), -1, dtype=int)
    for rank in range(10):
        players = np.unique(data["player"][data["y"] == rank])
        for index, player in enumerate(generator.permutation(players)):
            folds[data["player"] == player] = index % 5
    if (folds < 0).any():
        raise ValueError("Incomplete fold assignment")
    return folds


def cross_validate(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Refits identical C1+C2 calibration with each own-model replacement."""
    folds = game_folds(data)
    arrays: dict[str, list[np.ndarray]] = {"y": [], "player": [], "count": []}
    for fold in range(5):
        masks = (folds != fold, folds == fold)
        fitting = subset_data(data, masks[0])
        testing = subset_data(data, masks[1])
        if set(fitting["player"]) & set(testing["player"]):
            raise ValueError("Calibration split player overlap")
        questions, labels = residual_shift.validation_questions(
            testing["player"], testing["y"]
        )
        arrays["y"].append(labels)
        arrays["player"].append(
            np.asarray([testing["player"][rows[0]] for rows in questions])
        )
        arrays["count"].append(np.asarray([len(rows) for rows in questions]))
        for arm in ("original", *ARMS):
            for subset, mask in zip((fitting, testing), masks):
                subset["d96"] = (
                    data["d96"][mask]
                    if arm == "original"
                    else (data[arm].mean(axis=0)[mask])
                )
            model = calibration.fit(fitting, MEMBERS)
            arrays.setdefault(arm, []).append(
                calibration.predict(model, testing, questions)
            )
            if arm != "original":
                for seed, probabilities in enumerate(data[arm]):
                    testing["d96"] = probabilities[masks[1]]
                    arrays.setdefault(f"{arm}_seed{seed}", []).append(
                        calibration.predict(model, testing, questions)
                    )
        print(f"C1+C2 replacement evaluation fold {fold + 1}/5", flush=True)
    return {key: np.concatenate(parts) for key, parts in arrays.items()}


def subset_data(
    data: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    """Subsets game-level fields without mixing seed and game axes."""
    return {key: value[mask] for key, value in data.items() if key not in ARMS}


def summarize(arrays: dict[str, np.ndarray]) -> dict:
    """Reports official scores and player-clustered adjusted comparisons."""
    labels, players = arrays["y"], arrays["player"]
    methods = [key for key in arrays if key not in ("y", "player", "count")]
    rewards = {
        key: statistics.player_scores(arrays[key], labels) for key in methods
    }
    report: dict = {"scores": {}, "contrasts": {}, "scope": "exploratory"}
    for key in methods:
        report["scores"][key] = {
            "utility": float(rewards[key].mean()),
            "exact": float((rank_lib.decode(arrays[key]) == labels).mean()),
            "nll": float(
                -np.log(arrays[key][np.arange(len(labels)), labels]).mean()
            ),
            "by_rank": [
                float(rewards[key][labels == rank].mean()) for rank in range(10)
            ],
            "by_count": {
                str(count): float(rewards[key][arrays["count"] == count].mean())
                for count in np.unique(arrays["count"])
            },
        }
    for arm in ("nll_utility", "reward_utility"):
        difference = rewards[arm] - rewards["nll_nll"]
        clustered = np.asarray(
            [
                difference[players == player].mean()
                for player in np.unique(players)
            ]
        )
        seed_deltas = [
            float(
                (
                    rewards[f"{arm}_seed{seed}"]
                    - rewards[f"nll_nll_seed{seed}"]
                ).mean()
            )
            for seed in range(3)
        ]
        interval = statistics.bootstrap_players(clustered, confidence=0.975)
        worst_rank = min(
            report["scores"][arm]["by_rank"][rank]
            - report["scores"]["nll_nll"]["by_rank"][rank]
            for rank in range(10)
        )
        report["contrasts"][arm] = {
            "paired_player_97_5ci": interval,
            "seed_deltas_fixed_ensemble_calibration": seed_deltas,
            "worst_rank_delta": worst_rank,
            "passes_gate": interval["low"] > 0
            and sum(value > 0 for value in seed_deltas) >= 2
            and worst_rank >= -0.05,
        }
    return report


def main() -> None:
    """Requires all training artifacts before any outer evaluation."""
    args = train.arguments()
    if not (args.output / "TRAINING_COMPLETE").exists():
        raise RuntimeError("All training seeds must finish before evaluation")
    if (args.output / "report.json").exists():
        raise FileExistsError("Refusing to overwrite completed evaluation")
    torch.set_num_threads(4)
    arrays = cross_validate(load_data(args))
    report = summarize(arrays)
    np.savez_compressed(
        args.output / "evaluation.npz", allow_pickle=False, **arrays
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
