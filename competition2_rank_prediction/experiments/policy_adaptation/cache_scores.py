"""Caches frozen ten-profile legal likelihoods once for all matched pilot seeds.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
GPU1 inference only; no rank model fitting, evaluation utility or test inference.
"""

import hashlib
import json
import pathlib

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.policy_adaptation import features
from competition2_rank_prediction.experiments.policy_adaptation import probe


def extract(context: dict, payload: dict) -> np.ndarray:
    """Predicts all candidate log likelihoods without reading stored rank labels."""
    scores = []
    with torch.inference_mode():
        for start in range(0, len(payload["game_key"]), 4):
            rows = np.arange(start, min(start + 4, len(payload["game_key"])))
            values = features.game_scores(context, payload, rows).cpu().numpy()
            if values.shape != (len(rows), 10) or not np.isfinite(values).all():
                raise ValueError("Invalid frozen candidate scores")
            scores.append(values)
            if (start + 4) % 500 == 0:
                print(
                    f"Scored {start+4}/{len(payload['game_key'])} games",
                    flush=True,
                )
    return np.concatenate(scores)


def main() -> None:
    """Writes one isolated frozen-score cache and complete provenance manifest."""
    root = pathlib.Path.cwd()
    output = root / "production/policy_frozen_scores_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing frozen score cache: {output}")
    # Check completeness before loading or allocating the GPU model.
    training = features.load(root, "train")
    evaluation = features.load(root, "evaluation")
    if set(training["player_id"]) & set(evaluation["player_id"]):
        raise ValueError("Frozen-score training/evaluation player overlap")
    modules, payload = bridge.load_assets(
        root / "runtime/competition2_research"
    )
    context = {
        "model": bridge.build_model(modules, payload, "cuda:1", False),
        "profiles": probe.candidate_metadata(modules, 1, "cuda:1"),
        "device": "cuda:1",
    }
    output.mkdir(parents=True)
    for split, inputs in (("train", training), ("evaluation", evaluation)):
        values = extract(context, inputs)
        np.savez_compressed(
            output / f"{split}.npz",
            scores=values,
            game_key=inputs["game_key"],
            player_id=inputs["player_id"],
            rank=inputs["rank"],
            inverse_ranks=np.asarray(probe.INVERSE_RANKS),
        )
    paths = list(pathlib.Path(__file__).parent.glob("*.py"))
    paths.extend(
        (root / "production/policy_legal_inputs_20261109").rglob("*.npz")
    )
    paths.extend(
        (root / "production/human_finetune_inputs_20261109").rglob("*.npz")
    )
    paths.append(
        root
        / "runtime/competition2_research/models/human_trunk/b18c384nbt-humanv0.ckpt"
    )
    manifest = {
        str(path.relative_to(root)): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in paths
    }
    (output / "input_hashes.json").write_text(json.dumps(manifest, indent=2))
    (output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
