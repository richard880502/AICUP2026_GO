"""Extracts frozen HumanSL trunk tokens at each sampled own decision.

Example (repository root):
    PYTHONPATH=.:competition1_player_identification CUDA_VISIBLE_DEVICES=1 \
      python -m competition1_player_identification.experiments.trunk_style.extract \
      --games production/open_set_v2_20261111/engine_inputs/games.csv \
      --output production/trunk_style_20261113/corpus

Each game yields up to 64 evenly spaced own non-pass decisions, the same
sampler as the 64x41 engine cache. A token is the pretrained network's
384 global mean plus its 384 activations at the played point, taken before
the move, plus a pass flag. No identity or rank label enters the network.
Shards are written atomically and existing shards are skipped.
"""

import argparse
import collections
import concurrent.futures
import csv
import functools
import json
import multiprocessing
import pathlib
import time
import types
from typing import Any

import numpy as np
import torch

from competition2_rank_prediction import extract_rank_features
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import equivalence

SAMPLES = 64
DIMENSION = 769
SHARD = 1024
RESEARCH = pathlib.Path("runtime/competition2_research")
csv.field_size_limit(10**9)


@functools.lru_cache(maxsize=1)
def worker_modules() -> dict[str, types.ModuleType]:
    """Caches upstream CPU feature code inside each spawned worker."""
    torch.set_num_threads(1)
    return bridge.load_vendor(RESEARCH / "vendor/KataGo-human-v1.15.0/python")


def encode_job(config: dict, sgf: str, color: str) -> tuple[Any, np.ndarray]:
    """Builds network inputs for one game; malformed games return no rows."""
    try:
        moves = extract_rank_features.parse_moves(sgf)
        turns = extract_rank_features.katago.sample_player_turns(
            moves, color, SAMPLES
        )
        if not turns:
            return None, np.zeros(0, np.int64)
        inputs = equivalence.make_inputs(worker_modules(), config, moves, turns)
        points = np.asarray(
            [
                int(moves[turn].y or 0) * 19 + int(moves[turn].x or 0)
                for turn in turns
            ],
            np.int64,
        )
        return inputs, points
    except (ValueError, AssertionError, IndexError):
        return None, np.zeros(0, np.int64)


def infer(model: torch.nn.Module, modules: dict, batch: list) -> np.ndarray:
    """Returns [positions, 769] half-precision trunk tokens for queued games."""
    arrays = [
        torch.as_tensor(
            np.concatenate([item[0][column] for item in batch]), device="cuda:0"
        )
        for column in range(3)
    ]
    points = torch.as_tensor(
        np.concatenate([item[1] for item in batch]), device="cuda:0"
    )
    extra = modules["model_pytorch"].ExtraOutputs(["trunkfinal"])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        model(*arrays, extra_outputs=extra)
        trunk = extra.returned["trunkfinal"].float()
        local = trunk.flatten(start_dim=2).transpose(1, 2)[
            torch.arange(len(points), device="cuda:0"), points
        ]
        tokens = torch.cat(
            (trunk.mean(dim=(2, 3)), local, torch.zeros_like(local[:, :1])), 1
        )
    values = tokens.cpu().numpy().astype(np.float16)
    if values.shape[1] != DIMENSION or not np.isfinite(values).all():
        raise ValueError("Malformed or overflowing trunk tokens")
    return values


def extract_shard(pool: Any, context: tuple, games: list) -> tuple:
    """Encodes one shard with bounded outstanding CPU jobs and GPU batches."""
    model, modules, config = context
    tokens = np.zeros((len(games), SAMPLES, DIMENSION), np.float16)
    masks = np.zeros((len(games), SAMPLES), bool)
    pending: collections.deque = collections.deque()
    batch: list = []

    def flush() -> None:
        """Runs the queued games and stores each at its own row."""
        values = infer(model, modules, [item[1:] for item in batch])
        start = 0
        for index, _, points in batch:
            tokens[index, : len(points)] = values[start : start + len(points)]
            masks[index, : len(points)] = True
            start += len(points)
        batch.clear()

    def collect() -> None:
        """Moves the oldest finished CPU job into the GPU queue."""
        index, future = pending.popleft()
        inputs, points = future.result()
        if inputs is not None:
            batch.append((index, inputs, points))
        if len(batch) == 8:
            flush()

    for index, game in enumerate(games):
        pending.append(
            (
                index,
                pool.submit(
                    encode_job, config, game["sgf_content"], game["color"]
                ),
            )
        )
        if len(pending) >= 64:
            collect()
    while pending:
        collect()
    if batch:
        flush()
    return tokens, masks


def main() -> None:  # pylint: disable=too-many-locals
    """Extracts every missing shard of one game list in its file order."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Second worker: walk shards backwards and never mark completion.",
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    with arguments.games.open(encoding="utf-8", newline="") as source:
        games = list(csv.DictReader(source))
    if arguments.limit:
        games = games[: arguments.limit]
    modules, payload = bridge.load_assets(RESEARCH)
    model = bridge.build_model(modules, payload, "cuda:0", False)
    context = (model, modules, payload["config"])
    started = time.monotonic()
    done = 0
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=arguments.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        starts = list(range(0, len(games), SHARD))
        for start in reversed(starts) if arguments.reverse else starts:
            destination = arguments.output / f"shard_{start // SHARD:05d}.npz"
            if destination.exists():
                continue
            rows = games[start : start + SHARD]
            tokens, masks = extract_shard(pool, context, rows)
            temporary = destination.with_suffix(".tmp.npz")
            np.savez(
                temporary,
                tokens=tokens,
                masks=masks,
                record_ids=np.asarray([row["game_id"] for row in rows]),
            )
            temporary.rename(destination)
            done += len(rows)
            elapsed = time.monotonic() - started
            print(
                json.dumps(
                    {
                        "shard": destination.name,
                        "games_per_second": round(done / elapsed, 2),
                        "empty_games": int((~masks.any(1)).sum()),
                        "remaining_hours_if_single_worker": round(
                            (len(games) - start - len(rows))
                            / max(done / elapsed, 1e-6)
                            / 3600,
                            2,
                        ),
                    }
                ),
                flush=True,
            )
    if not arguments.reverse:
        (arguments.output / "COMPLETE").write_text(f"{len(games)} games\n")


if __name__ == "__main__":
    main()
