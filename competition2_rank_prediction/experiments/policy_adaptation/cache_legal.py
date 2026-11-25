# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Builds lossless legal-support caches aligned to existing HumanSL inputs.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Six CPU workers replay SGF/color only; rank labels never affect legal support.
"""

import concurrent.futures
import csv
import functools
import hashlib
import json
import multiprocessing
import pathlib
import sys

import numpy as np

from competition2_rank_prediction.experiments.human_trunk import extract
from competition2_rank_prediction.experiments.policy_adaptation import (
    equivalence,
)

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def encode(research: str, game: tuple[str, str]) -> dict:
    """Returns padded legal support and played positions without target metadata."""
    moves = extract_rank_features.parse_moves(game[0])
    turns = extract_rank_features.katago.sample_player_turns(moves, game[1], 8)
    if not turns:
        raise ValueError("No eligible decisions for legal-mask cache")
    legal = equivalence.legal_masks(
        extract.worker_modules(research), [(moves, turns)]
    )
    points = np.asarray(
        [
            361 if moves[turn].x is None else moves[turn].y * 19 + moves[turn].x
            for turn in turns
        ],
        dtype=np.int16,
    )
    if not legal[np.arange(len(points)), points].all():
        raise ValueError(
            "Played move is illegal in reconstructed training input"
        )
    rows = np.minimum(np.arange(8), len(turns) - 1)
    packed = np.packbits(legal[rows], axis=-1)
    if not np.array_equal(
        np.unpackbits(packed, axis=-1)[..., :362], legal[rows]
    ):
        raise ValueError("Legal mask packing changed support")
    return {
        "legal_packed": packed,
        "points": points[rows],
        "valid": np.arange(8) < len(turns),
    }


def write(
    pool, operation, games: list, source: pathlib.Path, destination
) -> None:
    """Requires exact alignment with previously verified spatial-input cache."""
    encoded = list(
        pool.map(
            operation,
            [(game["sgf_content"], game["color"]) for game in games],
            chunksize=8,
        )
    )
    values: dict = {
        name: np.stack([item[name] for item in encoded])
        for name in ("legal_packed", "points", "valid")
    }
    values["game_key"] = np.asarray(
        [game["game_key"] for game in games], dtype=str
    )
    with np.load(source) as inputs:
        for name in ("game_key", "points", "valid"):
            if not np.array_equal(values[name], np.asarray(inputs[name])):
                raise ValueError(
                    f"Spatial/legal cache alignment differs: {source}/{name}"
                )
    np.savez_compressed(destination, **values)
    print(f"Saved {destination}: {len(games)} games", flush=True)


def split_cache(pool, operation, context: tuple, manifest: dict) -> None:
    """Processes bounded500-game chunks, checking every source CSV hash."""
    root, split, source, output = context
    raw = root / "production/human_finetune_inputs_20261109"
    destination = output / split
    destination.mkdir()
    count = 0
    for path in sorted(source.glob("games_*.csv")):
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != manifest[str(path.relative_to(root))]
        ):
            raise ValueError(
                f"Source CSV changed since spatial input extraction: {path}"
            )
        with path.open(encoding="utf-8", newline="") as handle:
            games = list(csv.DictReader(handle))
        for start in range(0, len(games), 500):
            write(
                pool,
                operation,
                games[start : start + 500],
                raw / split / f"inputs_{count:04d}.npz",
                destination / f"legal_{count:04d}.npz",
            )
            count += 1


def main() -> None:
    """Creates masks only after completed spatial caches and engine parity."""
    root = pathlib.Path.cwd()
    output = root / "production/policy_legal_inputs_20261109"
    if output.exists():
        raise FileExistsError(f"Refusing existing legal input cache: {output}")
    if not (
        root / "production/human_finetune_inputs_20261109/COMPLETE"
    ).exists():
        raise ValueError("Original spatial input cache is incomplete")
    if not json.loads(
        (
            root / "production/policy_equivalence_20261109/report.json"
        ).read_text()
    )["passes_all_checks"]:
        raise ValueError("Ten-profile engine/support equivalence did not pass")
    manifest = json.loads(
        (
            root / "production/human_finetune_inputs_20261109/manifest.json"
        ).read_text()
    )
    sources = {
        "train": root / "production/game_coverage_manifest_20261109/cap8",
        "evaluation": root
        / "runtime/competition2_research/features/shards/val",
    }
    output.mkdir(parents=True)
    operation = functools.partial(
        encode, str(root / "runtime/competition2_research")
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=6, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for split, source in sources.items():
            split_cache(
                pool, operation, (root, split, source, output), manifest
            )
    (output / "COMPLETE").touch()


if __name__ == "__main__":
    main()
