"""Extracts frozen move-transformer embeddings for all training games.

The holdout-safe ``move_h.pt`` checkpoint excludes both the fixed validation
cohort and the 400-player holdout. Embeddings are streamed to a NumPy memmap
so extraction can resume without retaining a million vectors in memory.

Example:
    c1/.venv/bin/python -m \
      experiments.hierarchical_move.extract_embeddings --device cuda:0
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

import move_transformer
import rank_lib


def extract_batch(
    model: move_transformer.MoveTransformer,
    tokens: np.ndarray,
    lengths: np.ndarray,
    colours: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Encodes one batch using the canonical board orientation."""
    token_tensor = torch.from_numpy(tokens.astype(np.int32))
    length_tensor = torch.from_numpy(lengths.astype(np.int64))
    positions, move_colours, valid = move_transformer.unpack(
        token_tensor,
        length_tensor,
        device,
    )
    player_colours = torch.from_numpy(colours.astype(np.int64)).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeddings = model.encode(
            positions,
            move_colours,
            valid,
            player_colours,
        )
    return embeddings.cpu().numpy().astype(np.float16)


def output_array(
    path: pathlib.Path,
    rows: int,
    dimensions: int,
    resume: bool,
) -> np.memmap:
    """Creates or reopens the streaming embedding array."""
    if resume and path.exists():
        return np.lib.format.open_memmap(path, mode="r+")
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float16,
        shape=(rows, dimensions),
    )


def load_progress(path: pathlib.Path, resume: bool) -> int:
    """Returns the first unprocessed row from a progress file."""
    if not resume or not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8"))["completed_rows"])


def main() -> None:
    """Streams frozen game embeddings and aligned metadata to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moves",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "moves" / "all_moves512.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=rank_lib.ROOT / "results" / "move_h.pt",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "move_embeddings_h",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    with np.load(args.moves, allow_pickle=True) as data:
        tokens = data["tokens"][:, :256]
        lengths = np.minimum(data["length"], 256).astype(np.int16)
        colours = data["colour"].astype(np.int8)
        metadata = {
            key: data[key]
            for key in ("rank", "player_id", "game_key")
        }
    model = move_transformer.MoveTransformer().to(device)
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state"])
    model.eval()

    progress_path = args.output / "progress.json"
    start = load_progress(progress_path, args.resume)
    embeddings = output_array(
        args.output / "embeddings.npy",
        len(tokens),
        model.score.in_features,
        args.resume,
    )
    started = time.time()
    for row in range(start, len(tokens), args.batch):
        end = min(row + args.batch, len(tokens))
        embeddings[row:end] = extract_batch(
            model,
            tokens[row:end],
            lengths[row:end],
            colours[row:end],
            device,
        )
        if end % (args.batch * 20) == 0 or end == len(tokens):
            embeddings.flush()
            progress_path.write_text(
                json.dumps({"completed_rows": end}) + "\n",
                encoding="utf-8",
            )
            rate = (end - start) / max(time.time() - started, 1e-6)
            print(
                f"encoded {end}/{len(tokens)} games ({rate:.1f} games/s)",
                flush=True,
            )
    np.savez_compressed(
        args.output / "metadata.npz",
        colour=colours,
        **metadata,
    )
    manifest = {
        "games": len(tokens),
        "dimensions": model.score.in_features,
        "checkpoint": str(args.checkpoint),
        "moves": str(args.moves),
        "orientation_tta": 1,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
