"""P2 stage A, step 2: per-game probabilities of one fold's three-seed ensembles.

For each arm (``tenp3``, ``tenp10``) of cross-fit fold c, the checkpoints
``<arm>_c<c>_seed{0,1,2}.pt`` written by ``p2_train.sh`` score the validation
cohort and the holdout3 games.

The profile subset is read from each checkpoint's own ``keep_profiles``
argument and applied exactly as ``transformer_rank.load_splits`` does: the raw
``[games, decisions, 11, 3]`` tensor is sliced on its stream axis before
``decision_tokens``. Before anything is written for an arm, its validation
output is compared with the training script's own ``val_probs_<tag>.npz``; a
mismatch aborts the run.

Games without any valid decision get the uniform distribution, as in the
production cross-fit (``crossfit_cohort.evaluate.ckpt_probs``).

Example (research root):
    CUDA_VISIBLE_DEVICES=1 c1/.venv/bin/python -m \
        experiments.ten_profile_production.p2_score --fold 0 \
        --model-dir $P2/models --holdout3 $OUT/feat10_holdout3 \
        --output-dir $P2/probabilities
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

# Project modules of the research root (run with -m from there).
import rank_lib as lib  # pylint: disable=import-error
import transformer_rank as tr  # pylint: disable=import-error

SEEDS = 3
EXPECTED_PROFILES = {"tenp3": [0, 3, 5, 9], "tenp10": None}


def parse_args() -> argparse.Namespace:
    """Command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--fold", type=int, choices=(0, 1), required=True)
    parser.add_argument("--arms", default="tenp3,tenp10")
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--validation",
        type=pathlib.Path,
        default=pathlib.Path(
            "/mnt/share_data_77/richard/AICUP2026_GO/rankmle_audit/feat10_val"
        ),
    )
    parser.add_argument(
        "--holdout3",
        type=pathlib.Path,
        default=None,
        help="ten-profile holdout3 cache; omit to only verify the validation scores",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="max |p - p_train| allowed when reproducing val_probs_<tag>.npz",
    )
    return parser.parse_args()


def load_checkpoints(model_dir: pathlib.Path, tag: str) -> list[dict]:
    """The three seed checkpoints of one ensemble."""
    paths = [model_dir / f"{tag}_seed{seed}.pt" for seed in range(SEEDS)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete ensemble {tag}: {missing}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in paths]


def kept_profiles(checkpoints: list[dict], arm: str) -> list[int] | None:
    """The stream indices the ensemble was trained on (None = all eleven)."""
    values = {checkpoint["args"].get("keep_profiles") for checkpoint in checkpoints}
    if len(values) != 1:
        raise ValueError(f"{arm}: seeds disagree on keep_profiles {values}")
    value = values.pop()
    profiles = None if not value else [int(i) for i in value.split(",")]
    if profiles != EXPECTED_PROFILES[arm]:
        raise ValueError(f"{arm}: checkpoint keep_profiles {profiles} is not the P2 arm")
    return profiles


def ensemble_probabilities(
    features: dict[str, np.ndarray],
    checkpoints: list[dict],
    profiles: list[int] | None,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed-mean probabilities [G, 10] and the empty-game mask."""
    tokens = features["tokens"]
    if profiles is not None:
        tokens = tokens[:, :, profiles]  # identical to transformer_rank.load_splits
    x = tr.decision_tokens(tokens)
    valid = features["masks"].copy()
    empty: np.ndarray = np.asarray(~valid.any(axis=1))
    valid[empty, 0] = True
    predictions = []
    for checkpoint in checkpoints:
        settings = checkpoint["args"]
        if np.shape(checkpoint["mean"])[-1] != x.shape[2]:
            raise ValueError(
                f"checkpoint expects {np.shape(checkpoint['mean'])[-1]} channels, "
                f"features give {x.shape[2]}"
            )
        model = tr.RankTransformer(
            x.shape[2], x.shape[1], settings["d_model"], settings["heads"],
            settings["ffn"], settings["layers"], settings["dropout"],
            settings.get("head", "clm"),
        ).to(device)
        model.load_state_dict(checkpoint["state"])
        scaled = np.where(
            valid[..., None], (x - checkpoint["mean"]) / checkpoint["std"], 0
        ).astype(np.float32)
        predictions.append(tr.predict(model, scaled, valid, device))
    probabilities = np.mean(predictions, axis=0).astype(np.float64)
    probabilities[empty] = 0.1
    return probabilities, empty


def verify_validation(
    probabilities: np.ndarray,
    empty: np.ndarray,
    features: dict[str, np.ndarray],
    reference: pathlib.Path,
    tolerance: float,
) -> dict:
    """Compares the re-scored validation games with the training script's own output."""
    with np.load(reference, allow_pickle=True) as saved:
        keys = np.asarray(saved["game_key"]).astype(str)
        trained = np.asarray(saved["transformer"], dtype=np.float64)
        labels = np.asarray(saved["y"], dtype=np.int64)
    if not np.array_equal(keys, features["game_key"]):
        raise ValueError(f"{reference}: validation game order differs from the cache")
    if not np.array_equal(labels, lib.rank_ids(features["rank"])):
        raise ValueError(f"{reference}: validation labels differ from the cache")
    rows = ~empty  # the training script has no empty-game rule
    difference = float(np.abs(probabilities[rows] - trained[rows]).max())
    stats = {
        "reference": str(reference),
        "games": int(len(keys)),
        "empty_games": int(empty.sum()),
        "max_abs_diff": difference,
        "argmax_agreement": float(
            (probabilities[rows].argmax(1) == trained[rows].argmax(1)).mean()
        ),
    }
    if not difference <= tolerance:
        raise ValueError(f"scorer does not reproduce {reference}: {stats}")
    return stats


def main() -> None:
    """Scores, verifies and writes both arms of one fold."""
    args = parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(8)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"validation": lib.load_features(args.validation)}
    if args.holdout3 is not None:
        splits["holdout3"] = lib.load_features(args.holdout3)
    summary = {}
    for arm in args.arms.split(","):
        tag = f"{arm}_c{args.fold}"
        checkpoints = load_checkpoints(args.model_dir, tag)
        profiles = kept_profiles(checkpoints, arm)
        outputs = {}
        for name, features in splits.items():
            outputs[name] = ensemble_probabilities(features, checkpoints, profiles, device)
        check = verify_validation(
            *outputs["validation"],
            splits["validation"],
            args.model_dir / f"val_probs_{tag}.npz",
            args.tolerance,
        )
        summary[tag] = {"profiles": profiles, "validation_reproduction": check}
        for name, (probabilities, empty) in outputs.items():
            features = splits[name]
            np.savez_compressed(
                args.output_dir / f"{tag}_{name}.npz",
                game_key=features["game_key"],
                player_id=features["player_id"],
                rank=features["rank"],
                probabilities=probabilities,
                empty=empty,
            )
            labels = lib.rank_ids(features["rank"])
            summary[tag][name] = {
                "games": int(len(probabilities)),
                "empty_games": int(empty.sum()),
                "per_game_top1": float((probabilities.argmax(1) == labels).mean()),
            }
        print(json.dumps({tag: summary[tag]}, indent=2), flush=True)
    (args.output_dir / f"score_c{args.fold}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
