"""Broadens input-equivalence checks and benchmarks genuine spatial embeddings.

Example: python -m competition2_rank_prediction.experiments.human_trunk.benchmark
No rank labels enter the network or any accuracy experiment.
"""

import csv
import itertools
import json
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import equivalence
from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def prepare(
    args, modules, payload
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, ...], float]:
    """Encodes eight fixed games using exact upstream board/history code."""
    with (args.research / "features/shards/val/games_0000.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        games = list(itertools.islice(csv.DictReader(handle), 8))
    cases = []
    arrays: list[list[np.ndarray]] = [[], [], []]
    started = time.perf_counter()
    for game in games:
        moves = extract_rank_features.parse_moves(game["sgf_content"])
        turns = extract_rank_features.katago.sample_player_turns(
            moves, game["color"], 8
        )
        values = equivalence.make_inputs(
            modules, payload["config"], moves, turns
        )
        for destination, value in zip(arrays, values):
            destination.append(value)
        cases.append(
            {
                "game_key": game["game_key"],
                "color": game["color"],
                "moves": moves,
                "turns": turns,
            }
        )
    return (
        cases,
        tuple(np.concatenate(values) for values in arrays),
        time.perf_counter() - started,
    )


def infer(model, modules, arrays, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns full raw policy and global+played-point hidden representations."""
    extra = modules["model_pytorch"].ExtraOutputs(["trunkfinal"])
    with torch.inference_mode():
        output = model(*arrays[:3], extra_outputs=extra)
        trunk = extra.returned["trunkfinal"]
        global_features = trunk.mean(dim=(2, 3))
        spatial = trunk.flatten(start_dim=2).transpose(1, 2)
        points = arrays[3]
        local = spatial[
            torch.arange(len(points), device=device), points.clamp(max=360)
        ]
        local = torch.where((points < 361)[:, None], local, 0)
        embeddings = torch.cat(
            (global_features, local, (points == 361)[:, None]), dim=1
        )
        policy = output[0][0][:, 0].softmax(dim=-1)
    return policy.cpu().numpy(), embeddings.cpu().numpy()


def measure(args, modules, payload, cases, inputs) -> dict:
    """Measures GPU inference separately from Python board-feature construction."""
    model = bridge.build_model(modules, payload, args.device, False)
    points = []
    for case in cases:
        for turn in case["turns"]:
            move = case["moves"][turn]
            points.append(361 if move.x is None else move.y * 19 + move.x)
    arrays = [torch.as_tensor(values, device=args.device) for values in inputs]
    arrays.append(torch.as_tensor(points, device=args.device))
    infer(model, modules, arrays, args.device)
    torch.cuda.synchronize(device=args.device)
    torch.cuda.reset_peak_memory_stats(device=args.device)
    started = time.perf_counter()
    for _ in range(10):
        policy, embeddings = infer(model, modules, arrays, args.device)
    torch.cuda.synchronize(device=args.device)
    elapsed = time.perf_counter() - started
    if (
        embeddings.shape != (len(points), 769)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Invalid global+played-point hidden embedding")
    return {
        "positions": len(points),
        "games": len(cases),
        "colors": sorted(set(case["color"] for case in cases)),
        "embedding_shape": list(embeddings.shape),
        "embedding_definition": (
            "384 global mean +384 played point +1 pass indicator; "
            "neutral1d metadata"
        ),
        "gpu_positions_per_second": len(points) * 10 / elapsed,
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(
            device=args.device
        ),
        "policy_equivalence": equivalence.compare(
            policy, references(args, cases)
        ),
        "scope": "fixed small-sample feasibility/throughput, not rank utility",
    }


def references(args, cases) -> np.ndarray:
    """Queries fixed engine policies in the same game/turn order as the batch."""
    return np.concatenate(
        [
            equivalence.engine_policies(args, case["moves"], case["turns"])
            for case in cases
        ]
    )


def main() -> None:
    """Writes a non-overwriting benchmark report for the verified raw weights."""
    args = common.parse_arguments(__doc__ or "", "human_trunk_20261109", True)
    if args.device != "cuda:1":
        raise ValueError(
            "Benchmark uses GPU1 only; GPU0 service is out of scope"
        )
    destination = args.output / "benchmark.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {destination}")
    modules, payload = bridge.load_assets(args.research)
    cases, inputs, feature_time = prepare(args, modules, payload)
    report = measure(args, modules, payload, cases, inputs)
    report["cpu_feature_seconds"] = feature_time
    report["cpu_feature_positions_per_second"] = (
        report["positions"] / feature_time
    )
    args.output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
