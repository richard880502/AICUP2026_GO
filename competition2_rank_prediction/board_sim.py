"""Fast 19x19 Go board simulation (numba) producing the official baseline's board-state history.

states[0] is the empty board (the SGF root); states[i] is the board after move i. Values: 0 empty,
1 black, 2 white. Captures follow standard rules; a suicide move removes the played group.
"""
import numpy as np
from numba import njit

N = 19


@njit(cache=True)
def _group(board, y, x, seen, stack, members):
    """Flood-fills the group at (y, x); returns (size, liberty count)."""
    colour = board[y, x]
    size = 0
    libs = 0
    top = 0
    stack[0] = y * N + x
    top = 1
    seen[y, x] = True
    # liberties are counted once per empty neighbour point via a stamp stored in `seen`
    while top > 0:
        top -= 1
        p = stack[top]
        cy, cx = p // N, p % N
        members[size] = p
        size += 1
        for d in range(4):
            ny = cy + (-1 if d == 0 else (1 if d == 1 else 0))
            nx = cx + (-1 if d == 2 else (1 if d == 3 else 0))
            if ny < 0 or ny >= N or nx < 0 or nx >= N:
                continue
            v = board[ny, nx]
            if v == 0:
                if not seen[ny, nx]:
                    seen[ny, nx] = True
                    libs += 1
            elif v == colour and not seen[ny, nx]:
                seen[ny, nx] = True
                stack[top] = ny * N + nx
                top += 1
    return size, libs


@njit(cache=True)
def _clear_seen(seen, members, size, board):
    for i in range(size):
        p = members[i]
        y, x = p // N, p % N
        seen[y, x] = False
        for d in range(4):
            ny = y + (-1 if d == 0 else (1 if d == 1 else 0))
            nx = x + (-1 if d == 2 else (1 if d == 3 else 0))
            if 0 <= ny < N and 0 <= nx < N:
                seen[ny, nx] = False


@njit(cache=True)
def simulate(pos, colour, count):
    """pos[i] in 0..361 (361 = pass), colour[i] in {0 black, 1 white}. Returns int8 [count + 1, 19, 19]."""
    states = np.zeros((count + 1, N, N), np.int8)
    board = np.zeros((N, N), np.int8)
    seen = np.zeros((N, N), np.bool_)
    stack = np.zeros(N * N, np.int64)
    members = np.zeros(N * N, np.int64)
    for i in range(count):
        p = pos[i]
        if p < 361:
            y, x = p // N, p % N
            me = np.int8(1 + colour[i])
            foe = np.int8(3 - me)
            if board[y, x] == 0:
                board[y, x] = me
                for d in range(4):
                    ny = y + (-1 if d == 0 else (1 if d == 1 else 0))
                    nx = x + (-1 if d == 2 else (1 if d == 3 else 0))
                    if 0 <= ny < N and 0 <= nx < N and board[ny, nx] == foe:
                        size, libs = _group(board, ny, nx, seen, stack, members)
                        _clear_seen(seen, members, size, board)
                        if libs == 0:
                            for k in range(size):
                                board[members[k] // N, members[k] % N] = 0
                size, libs = _group(board, y, x, seen, stack, members)
                _clear_seen(seen, members, size, board)
                if libs == 0:  # suicide
                    for k in range(size):
                        board[members[k] // N, members[k] % N] = 0
        states[i + 1] = board
    return states


def history_planes(tokens: np.ndarray, length: int, history: int = 250) -> np.ndarray:
    """int8 [history, 19, 19]: the last `history` board states, front-padded with empty boards."""
    pos = (tokens[:length].astype(np.int64) & 0x1FF)
    pos = np.minimum(pos, 361)
    colour = (tokens[:length].astype(np.int64) >> 9) & 1
    states = simulate(pos, colour, length)
    out = np.zeros((history, N, N), np.int8)
    take = states[-history:]
    out[history - len(take):] = take
    return out
