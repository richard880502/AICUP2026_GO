"""Builds the verified two-sided Transformer inputs for submission inference.

Matches the historical joint_game experiment's side and move-time channels.
"""

import argparse
import pathlib

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error


def fractions(keys: np.ndarray, moves_path: pathlib.Path) -> np.ndarray:
    """Returns own/opponent sampled move fractions with shape2 x games x96."""
    with np.load(moves_path) as saved:
        moves = {
            name: np.asarray(saved[name])
            for name in ("game_key", "tokens", "length", "colour")
        }
    lookup = {str(key): index for index, key in enumerate(moves["game_key"])}
    result = np.zeros((2, len(keys), 96), dtype=np.float32)
    for game, key in enumerate(keys):
        row = lookup[str(key)]
        length = int(moves["length"][row])
        tokens = moves["tokens"][row, :length].astype(np.int64)
        result[:, game] = _game_fractions(
            tokens, length, int(moves["colour"][row])
        )
    return result


def _game_fractions(tokens: np.ndarray, length: int, colour: int) -> np.ndarray:
    result = np.zeros((2, 96), dtype=np.float32)
    mover = (tokens >> 9) & 1
    passed = (tokens & 0x1FF) == 361
    for side, player_colour in enumerate((colour, 1 - colour)):
        turns = np.flatnonzero((mover == player_colour) & ~passed)
        if len(turns):
            selected = turns[
                np.linspace(0, len(turns) - 1, min(96, len(turns)), dtype=int)
            ]
            result[side, : len(selected)] = selected / max(length, 1)
    return result


def inputs(
    own: dict[str, np.ndarray],
    opponent: dict[str, np.ndarray],
    moves_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenates96 own and96 opponent tokens, including side and time."""
    lookup = {str(key): index for index, key in enumerate(opponent["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in own["game_key"]])
    timing = fractions(own["game_key"], moves_path)
    own_tokens = transformer_rank.decision_tokens(own["tokens"])
    opponent_tokens = transformer_rank.decision_tokens(opponent["tokens"][rows])
    side = np.zeros(own_tokens.shape[:2] + (1,), dtype=np.float32)
    tokens = np.concatenate(
        (
            np.concatenate((own_tokens, side, timing[0, ..., None]), axis=2),
            np.concatenate(
                (opponent_tokens, side + 1, timing[1, ..., None]), axis=2
            ),
        ),
        axis=1,
    )
    masks = np.concatenate((own["masks"], opponent["masks"][rows]), axis=1)
    return tokens.astype(np.float32), masks


def predict(
    args: argparse.Namespace, split: str, keys: np.ndarray
) -> np.ndarray:
    """Predicts the three newly trained full-training joint checkpoints."""
    if split == "validation":
        own_dir = args.research / "features/feat96/val"
        opponent_dir = args.research / "features/feat96opp/val"
        moves = args.research / "features/moves/all_moves512.npz"
    else:
        own_dir = args.baseline / "features/player"
        opponent_dir = args.baseline / "features/opponent"
        moves = args.baseline / "moves.npz"
    own = rank_lib.load_features(own_dir)
    lookup = {str(key): row for row, key in enumerate(own["game_key"])}
    rows = np.asarray([lookup[str(key)] for key in keys])
    own = {name: values[rows] for name, values in own.items()}
    tokens, masks = inputs(own, rank_lib.load_features(opponent_dir), moves)
    predictions = []
    paths = sorted(
        (args.research / "results").glob("joint_submit20261108_seed*.pt")
    )
    if len(paths) != 3 or not masks.any(axis=1).all():
        raise ValueError(
            "Joint prediction needs three checkpoints and usable games"
        )
    for path in paths:
        predictions.append(
            checkpoint_prediction(
                path, tokens, masks, torch.device(args.device)
            ),
        )
    return np.mean(predictions, axis=0)


def checkpoint_prediction(
    path: pathlib.Path,
    tokens: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Predicts a decision checkpoint using its own training normalization."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    normalization = saved.get("normalization", saved)
    scaled = np.where(
        masks[..., None],
        (tokens - normalization["mean"]) / normalization["std"],
        0,
    ).astype(np.float32)
    model = transformer_rank.RankTransformer(
        tokens.shape[2], tokens.shape[1]
    ).to(device)
    model.load_state_dict(saved["state"])
    return transformer_rank.predict(model, scaled, masks, device, 256)
