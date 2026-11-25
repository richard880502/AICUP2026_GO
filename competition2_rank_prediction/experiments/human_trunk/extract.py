# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Extracts frozen HumanSL global/played-point embeddings with CPU workers.

Example: python -m competition2_rank_prediction.experiments.human_trunk.extract
Eight spawned feature workers feed GPU1; originals and official tests are read-only.
"""

import argparse
import collections
import concurrent.futures
import csv
import dataclasses
import functools
import hashlib
import json
import multiprocessing
import pathlib
import sys
import types
from typing import Any

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import benchmark
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import equivalence

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error

SAMPLES = 16
DIMENSION = 769


@dataclasses.dataclass
class Context:
    """Carries frozen inference resources only, not labels.

    Attributes:
        model: Frozen pretrained model on GPU1.
        modules: Pinned upstream module objects.
        config: Original model architecture/feature configuration.
        research: Runtime input/vendor root.
        device: Explicit inference device.
    """

    model: torch.nn.Module
    modules: dict[str, types.ModuleType]
    config: dict[str, Any]
    research: pathlib.Path
    device: str


def arguments() -> argparse.Namespace:
    """Returns isolated resource/output paths and bounded worker count."""
    root = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--research",
        type=pathlib.Path,
        default=root / "runtime/competition2_research",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=root / "production/human_trunk_20261109",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


@functools.lru_cache(maxsize=1)
def worker_modules(research: str) -> dict[str, types.ModuleType]:
    """Caches only upstream CPU feature code within each spawned worker."""
    torch.set_num_threads(1)
    return bridge.load_vendor(
        pathlib.Path(research) / "vendor/KataGo-human-v1.15.0/python"
    )


def encode_job(
    research: str, config: dict, sgf: str, color: str
) -> tuple[Any, np.ndarray, str]:
    """Encodes SGF/color only; no actual rank/player metadata enters the worker."""
    try:
        moves = extract_rank_features.parse_moves(sgf)
        turns = extract_rank_features.katago.sample_player_turns(
            moves, color, SAMPLES
        )
        if not turns:
            return None, np.zeros(0, dtype=np.int64), "no eligible decisions"
        inputs = equivalence.make_inputs(
            worker_modules(research), config, moves, turns
        )
        points = np.asarray(
            [
                (
                    361
                    if moves[turn].x is None
                    else moves[turn].y * 19 + moves[turn].x
                )
                for turn in turns
            ],
            dtype=np.int64,
        )
        return inputs, points, ""
    except (ValueError, AssertionError, IndexError) as error:
        return (
            None,
            np.zeros(0, dtype=np.int64),
            f"{type(error).__name__}: {error}",
        )


def ordered_jobs(pool, context: Context, games: list[dict[str, str]]):
    """Yields ordered results while bounding outstanding feature jobs to sixteen."""
    pending: collections.deque[concurrent.futures.Future[Any]] = (
        collections.deque()
    )
    for game in games:
        pending.append(
            pool.submit(
                encode_job,
                str(context.research),
                context.config,
                game["sgf_content"],
                game["color"],
            )
        )
        if len(pending) >= 16:
            yield pending.popleft().result()
    while pending:
        yield pending.popleft().result()


def infer_batch(context: Context, batch: list) -> list[np.ndarray]:
    """Runs at most four variable-length games using the benchmarked GPU path."""
    inputs = []
    for column in range(3):
        inputs.append(np.concatenate([item[1][column] for item in batch]))
    points = np.concatenate([item[2] for item in batch])
    arrays = [torch.as_tensor(value, device=context.device) for value in inputs]
    arrays.append(torch.as_tensor(points, device=context.device))
    _, embeddings = benchmark.infer(
        context.model, context.modules, arrays, context.device
    )
    if (
        embeddings.shape != (len(points), DIMENSION)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("Malformed pretrained embeddings")
    pieces = []
    start = 0
    for item in batch:
        count = len(item[2])
        stored = embeddings[start : start + count].astype(np.float16)
        if not np.isfinite(stored).all():
            raise ValueError("Pretrained embedding overflows fp16 storage")
        pieces.append(stored)
        start += count
    return pieces


def extract_shard(
    pool, context: Context, games: list[dict[str, str]]
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns fp16embeddings and masks while recording invalid input games."""
    tokens = np.zeros((len(games), SAMPLES, DIMENSION), dtype=np.float16)
    masks = np.zeros((len(games), SAMPLES), dtype=bool)
    errors = [""] * len(games)
    batch = []
    for index, (inputs, points, error) in enumerate(
        ordered_jobs(pool, context, games)
    ):
        errors[index] = error
        if inputs is not None:
            batch.append((index, inputs, points))
        if len(batch) == 4:
            fill_batch(context, batch, tokens, masks)
            batch.clear()
        if (index + 1) % 128 == 0:
            print(
                f"encoded {index + 1}/{len(games)} games in shard", flush=True
            )
    if batch:
        fill_batch(context, batch, tokens, masks)
    return tokens, masks, errors


def fill_batch(
    context: Context, batch: list, tokens: np.ndarray, masks: np.ndarray
) -> None:
    """Places inference results back at immutable input-game row positions."""
    for item, embeddings in zip(batch, infer_batch(context, batch)):
        index = item[0]
        tokens[index, : len(embeddings)] = embeddings
        masks[index, : len(embeddings)] = True


def manifest(args: argparse.Namespace) -> None:
    """Refuses mixed-source/model resume and records fixed inference settings."""
    source = pathlib.Path(__file__).resolve().parent
    settings = {
        "samples": SAMPLES,
        "dimension": DIMENSION,
        "profile": "rank_1d",
        "weights": "raw",
        "sources": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in (
                "extract.py",
                "bridge.py",
                "equivalence.py",
                "smoke.py",
                "benchmark.py",
            )
        },
    }
    checkpoint = args.research / "models/human_trunk/b18c384nbt-humanv0.ckpt"
    with checkpoint.open("rb") as handle:
        settings["checkpoint_sha256"] = hashlib.file_digest(
            handle, "sha256"
        ).hexdigest()
    if (
        settings["checkpoint_sha256"]
        != "84b23476b56a73a775a0f09bad7d1c5c435690adb343aa851e80bf198c0c8b27"
    ):
        raise ValueError("Unexpected pretrained checkpoint content hash")
    destination = args.output / "extraction_manifest.json"
    encoded = json.dumps(settings, indent=2)
    if destination.exists() and destination.read_text() != encoded:
        raise ValueError(
            "Incompatible source/model settings for resumed extraction"
        )
    destination.write_text(encoded)


def process_shards(pool, context: Context, output: pathlib.Path) -> None:
    """Extracts original train/val inputs, with atomic independent shard writes."""
    for split in ("train", "val"):
        directory = output / "features" / split
        directory.mkdir(parents=True, exist_ok=True)
        for shard in sorted(
            (context.research / f"features/shards/{split}").glob("games_*.csv")
        ):
            destination = directory / shard.name.replace(
                "games_", "feat_"
            ).replace(".csv", ".npz")
            if destination.exists():
                continue
            with shard.open(encoding="utf-8", newline="") as handle:
                games = list(csv.DictReader(handle))
            tokens, masks, errors = extract_shard(pool, context, games)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                tokens=tokens,
                masks=masks,
                errors=np.asarray(errors),
                game_key=np.asarray([game["game_key"] for game in games]),
                player_id=np.asarray([game["player_id"] for game in games]),
                rank=np.asarray([game["rank"] for game in games]),
            )
            temporary.rename(destination)
            print(
                f"saved {split}/{destination.name}: {len(games)} games, "
                f"invalid={sum(bool(error) for error in errors)}",
                flush=True,
            )


def main() -> None:
    """Uses spawned CPU workers and GPU1 only, with no neural training."""
    args = arguments()
    if args.device != "cuda:1" or not 1 <= args.workers <= 16:
        raise ValueError("Use GPU1 and between one and sixteen CPU workers")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest(args)
    modules, payload = bridge.load_assets(args.research)
    context = Context(
        bridge.build_model(modules, payload, args.device, False),
        modules,
        payload["config"],
        args.research,
        args.device,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        process_shards(pool, context, args.output)
    (args.output / "EXTRACTION_COMPLETE").write_text(
        "Train and validation shards completed.\n"
    )


if __name__ == "__main__":
    main()
