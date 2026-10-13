"""SGF move parsing and fixed-length token construction.

Example:
    tokens, length = tokenize_sgf("(;SZ[19];B[pd];W[dd])", 512)
"""

import re

import numpy as np


BOARD_SIZE = 19
PASS_COORDINATE = BOARD_SIZE
PADDING_COORDINATE = BOARD_SIZE + 1
PADDING_COLOR = 2
MOVE_PATTERN = re.compile(r";([BW])\[([a-s]{0,2})\]")


def tokenize_sgf(sgf_content: str, max_moves: int) -> tuple[np.ndarray, int]:
    """Converts 19x19 SGF moves to x, y, color, and move-index tokens.

    Args:
        sgf_content: A complete SGF record containing ``;B[xy]`` moves.
        max_moves: Maximum number of moves retained from the game.

    Returns:
        A ``[max_moves, 4]`` int64 token array and its unpadded length.

    Raises:
        ValueError: If ``max_moves`` is not positive.
    """
    if max_moves <= 0:
        raise ValueError(f"max_moves must be positive, got {max_moves!r}")

    tokens = np.empty((max_moves, 4), dtype=np.int64)
    tokens[:, 0:2] = PADDING_COORDINATE
    tokens[:, 2] = PADDING_COLOR
    tokens[:, 3] = 0
    moves = MOVE_PATTERN.findall(sgf_content)
    length = min(len(moves), max_moves)
    for move_index, (color, coordinate) in enumerate(moves[:max_moves]):
        tokens[move_index, 2] = 0 if color == "B" else 1
        tokens[move_index, 3] = move_index
        if coordinate:
            tokens[move_index, 0] = ord(coordinate[0]) - ord("a")
            tokens[move_index, 1] = ord(coordinate[1]) - ord("a")
        else:
            tokens[move_index, 0:2] = PASS_COORDINATE
    return tokens, length
