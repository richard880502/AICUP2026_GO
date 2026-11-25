"""Prepares aligned mining and frozen C1+C2 probabilities for a new submission.

Run the companion run module after preparation completes.
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import residual_shift  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.mining_submission import (
    joint_features,
)

MEMBERS = ("d96", "resnet8c", "opp_scratch", "mining")


def arguments() -> argparse.Namespace:
    """Returns explicit read-only inputs and an isolated output directory."""
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research",
        type=pathlib.Path,
        default=repository / "runtime/competition2_research",
    )
    parser.add_argument(
        "--baseline",
        type=pathlib.Path,
        default=repository / "production/rank_20261030",
    )
    parser.add_argument(
        "--mining",
        type=pathlib.Path,
        default=repository / "production/semi_hard_mining_direct_20261108",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=repository / "production/rank_mining_20261108",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--member", choices=("mining", "joint"), default="mining"
    )
    return parser.parse_args()


def _validation(root: pathlib.Path) -> dict[str, np.ndarray]:
    result = {}
    for index, name in enumerate(MEMBERS[:3]):
        with np.load(
            root / f"results/val_probs_{name}.npz", allow_pickle=True
        ) as payload:
            if index == 0:
                result = {
                    key: np.asarray(payload[key])
                    for key in ("y", "player", "game_key")
                }
            lookup = {
                str(key): row for row, key in enumerate(payload["game_key"])
            }
            rows = np.asarray([lookup[str(key)] for key in result["game_key"]])
            field = "resnet8c" if name == "resnet8c" else "transformer"
            result[name] = np.asarray(payload[field])[rows]
            if not np.array_equal(np.asarray(payload["y"])[rows], result["y"]):
                raise ValueError(f"Validation labels differ for {name}")
    final = residual_shift.load_final(root / "features/final_result/val")
    result["final"] = np.stack([final[str(key)] for key in result["game_key"]])
    return result


def _test(directory: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(directory / "per_game_predictions.npz") as payload:
        result = {name: np.asarray(payload[name]) for name in payload.files}
    final = residual_shift.load_final(directory / "features/final")
    result["final"] = np.stack([final[str(key)] for key in result["game_key"]])
    return result


def _predict(
    context: tuple[pathlib.Path, torch.device],
    features: dict[str, np.ndarray],
    keys: np.ndarray,
) -> np.ndarray:
    directory, device = context
    lookup = {str(key): row for row, key in enumerate(features["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in keys])
    tokens = transformer_rank.decision_tokens(features["tokens"][rows])
    valid = features["masks"][rows].copy()
    empty = ~valid.any(axis=1)
    if empty.any():
        raise ValueError("Mining input has games with no usable decisions")
    predictions = []
    paths = sorted(directory.glob("seed_*/C.pt"))
    if len(paths) != 3:
        raise ValueError("Exactly three mining checkpoints are required")
    for path in paths:
        predictions.append(
            joint_features.checkpoint_prediction(path, tokens, valid, device)
        )
    return np.mean(predictions, axis=0)


def main() -> None:
    """Checks validation exclusion and prepares reproducible input caches."""
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    torch.set_num_threads(4)
    validation = _validation(args.research)
    test = _test(args.baseline)
    if args.member == "mining":
        with np.load(args.mining / "splits.npz") as split:
            training_players = set(split["train"])
    else:
        training_players = set(
            rank_lib.load_features(args.research / "features/feat96/train")[
                "player_id"
            ]
        )
    overlap = set(validation["player"]) & training_players
    if overlap:
        raise ValueError(
            f"Candidate training includes {len(overlap)} validation players"
        )
    device = torch.device(args.device)
    for name, values, directory in (
        ("validation", validation, args.research / "features/feat96/val"),
        ("test", test, args.baseline / "features/player"),
    ):
        if args.member == "mining":
            values[args.member] = _predict(
                (args.mining, device),
                rank_lib.load_features(directory),
                values["game_key"],
            )
        else:
            values[args.member] = joint_features.predict(
                args, name, values["game_key"]
            )
        print(
            f"{name}: {len(values['game_key'])} {args.member} predictions",
            flush=True,
        )
    args.output.mkdir(parents=True)
    arrays = {}
    for name, values in (("val", validation), ("test", test)):
        for key, value in values.items():
            arrays[f"{name}_{key}"] = value
    np.savez_compressed(
        args.output / "prepared.npz", allow_pickle=False, **arrays
    )
    manifest = _manifest(args, validation, len(overlap))
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _manifest(
    args: argparse.Namespace,
    validation: dict[str, np.ndarray],
    overlap: int,
) -> dict[str, object]:
    paths = (
        sorted(args.mining.glob("seed_*/C.pt"))
        if args.member == "mining"
        else sorted(
            (args.research / "results").glob("joint_submit20261108_seed*.pt")
        )
    )
    return {
        "members": list(MEMBERS[:3]) + [args.member],
        "validation_players": len(set(validation["player"])),
        "candidate_training_validation_overlap": overlap,
        "baseline_sha256": hashlib.sha256(
            (args.baseline / "submission_rank.csv").read_bytes()
        ).hexdigest(),
        "checkpoints_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "official_test_zip_sha256": hashlib.sha256(
            (
                args.research
                / "data/aicup2026_rank_prediction_tests_20261030.zip"
            ).read_bytes(),
        ).hexdigest(),
    }


if __name__ == "__main__":
    main()
