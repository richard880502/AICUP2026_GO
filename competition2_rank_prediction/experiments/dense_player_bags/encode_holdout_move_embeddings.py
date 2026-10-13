"""Encodes official-format holdout moves with the frozen raw-move encoder.

The official query archive has no rank column, unlike the training-move
archive. This narrow encoder only needs move tokens, lengths, player colour,
and game keys, then writes embeddings compatible with `attach_move_embeddings`.

Example:
    c1/.venv/bin/python -m \
        experiments.dense_player_bags.encode_holdout_move_embeddings --device cpu
"""

import argparse
import dataclasses
import json
import pathlib

import numpy as np
import torch

import move_transformer
import rank_lib


@dataclasses.dataclass(frozen=True)
class HoldoutMoves:
    """Contains the rank-free inputs and metadata of official query games."""

    tokens: np.ndarray
    lengths: np.ndarray
    colours: np.ndarray
    game_keys: np.ndarray
    players: np.ndarray


def encode_batch(
    model: move_transformer.MoveTransformer,
    tokens: np.ndarray,
    lengths: np.ndarray,
    colours: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Returns raw-move embeddings for one batch of query games."""
    positions, move_colours, valid = move_transformer.unpack(
        torch.from_numpy(tokens.astype(np.int32)),
        torch.from_numpy(lengths.astype(np.int64)),
        device,
    )
    player_colours = torch.from_numpy(colours.astype(np.int64)).to(device)
    with torch.no_grad(), torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        embeddings = model.encode(
            positions,
            move_colours,
            valid,
            player_colours,
        )
    return embeddings.cpu().numpy().astype(np.float16)


def load_moves(path: pathlib.Path) -> HoldoutMoves:
    """Loads rank-free query moves and limits their encoder context length."""
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    return HoldoutMoves(
        tokens=np.asarray(payload["tokens"])[:, :256],
        lengths=np.minimum(
            np.asarray(payload["length"]),
            256,
        ).astype(np.int16),
        colours=np.asarray(payload["colour"]).astype(np.int8),
        game_keys=np.asarray(payload["game_key"], dtype=str),
        players=np.asarray(payload["player_id"], dtype=str),
    )


def load_model(
    checkpoint_path: pathlib.Path,
    device: torch.device,
) -> move_transformer.MoveTransformer:
    """Restores the holdout-safe frozen raw-move encoder."""
    model = move_transformer.MoveTransformer().to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return model


def main() -> None:
    """Writes frozen move embeddings for rank-unknown holdout queries."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moves",
        type=pathlib.Path,
        default=rank_lib.ROOT / "features" / "moves" / "holdout_moves.npz",
    )
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=rank_lib.ROOT / "results" / "move_h.pt",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=(
            rank_lib.ROOT
            / "experiments"
            / "dense_player_bags"
            / "move_holdout_cpu"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=32)
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(arguments.device)
    moves = load_moves(arguments.moves)
    model = load_model(arguments.checkpoint, device)
    parts = []
    for start in range(0, len(moves.tokens), arguments.batch):
        end = min(start + arguments.batch, len(moves.tokens))
        parts.append(
            encode_batch(
                model,
                moves.tokens[start:end],
                moves.lengths[start:end],
                moves.colours[start:end],
                device,
            ),
        )
        if end % (arguments.batch * 20) == 0 or end == len(moves.tokens):
            print(f"encoded {end}/{len(moves.tokens)}", flush=True)
    embeddings = np.concatenate(parts)
    np.save(arguments.output / "embeddings.npy", embeddings)
    np.savez_compressed(
        arguments.output / "metadata.npz",
        colour=moves.colours,
        game_key=moves.game_keys,
        player_id=moves.players,
    )
    manifest = {
        "games": len(moves.tokens),
        "dimensions": embeddings.shape[1],
        "checkpoint": str(arguments.checkpoint),
        "moves": str(arguments.moves),
    }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
