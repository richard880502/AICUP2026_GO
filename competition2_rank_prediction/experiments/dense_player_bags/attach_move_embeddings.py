"""Appends cached raw-move embeddings to dense KataGo/HumanSL vectors.

`move_embeddings_h` was encoded from raw board-coordinate sequences for all
training games. It supplies a style signal that is complementary to the
KataGo/HumanSL decision representation and does not require a new GPU pass.

Example:
    c1/.venv/bin/python -m \
        experiments.dense_player_bags.attach_move_embeddings
"""

import argparse
import pathlib

import numpy as np

import rank_lib

DEFAULT_SPLITS = ("train", "validation")


def load_move_lookup(
    directory: pathlib.Path,
) -> tuple[dict[str, int], np.ndarray]:
    """Loads game-key rows and memory-mapped frozen move embeddings."""
    with np.load(directory / "metadata.npz", allow_pickle=True) as data:
        game_keys = np.asarray(data["game_key"], dtype=str)
    lookup = {key: index for index, key in enumerate(game_keys)}
    return lookup, np.load(directory / "embeddings.npy", mmap_mode="r")


def aligned_move_embeddings(
    keys: np.ndarray,
    lookup: dict[str, int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Returns cached move representations in the provided game-key order."""
    rows = []
    missing = []
    for key in keys.astype(str):
        row = lookup.get(key)
        if row is None:
            missing.append(key)
        else:
            rows.append(row)
    if missing:
        example = ", ".join(missing[:3])
        raise ValueError(
            f"Move embeddings lack {len(missing)} games; examples: {example}",
        )
    return embeddings[np.asarray(rows, dtype=np.int64)].astype(np.float16)


def append_split(
    name: str,
    source: pathlib.Path,
    output: pathlib.Path,
    lookup: dict[str, int],
    move_embeddings: np.ndarray,
) -> None:
    """Writes one dense split after appending aligned move representations."""
    with np.load(source / f"{name}.npz", allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    moves = aligned_move_embeddings(
        payload["game_key"],
        lookup,
        move_embeddings,
    )
    payload["embeddings"] = np.concatenate(
        (payload["embeddings"].astype(np.float16), moves),
        axis=1,
    )
    np.savez_compressed(output / f"{name}.npz", **payload)
    print(f"saved {name}: {payload['embeddings'].shape}", flush=True)


def main() -> None:
    """Creates dense embeddings augmented with frozen raw-move features."""
    parser = argparse.ArgumentParser()
    experiment = pathlib.Path(__file__).resolve().parent
    parser.add_argument(
        "--source",
        type=pathlib.Path,
        default=experiment / "embeddings_logits",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=experiment / "embeddings_logits_moves",
    )
    parser.add_argument(
        "--move-embeddings",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "move_embeddings_h",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "holdout"),
        default=DEFAULT_SPLITS,
    )
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    lookup, move_embeddings = load_move_lookup(arguments.move_embeddings)
    for name in arguments.splits:
        append_split(
            name,
            arguments.source,
            arguments.output,
            lookup,
            move_embeddings,
        )


if __name__ == "__main__":
    main()
