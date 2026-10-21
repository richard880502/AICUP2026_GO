"""Tactical "fighting context" features for the same decisions as the dense KataGo features.

No engine is needed. For every sampled decision of the labelled player (identical sampling to
extract_rank_features.py) the board before and after the move is simulated (board_sim.py) and 10
numbers are recorded:
  0 adjacent opponent stones       1 adjacent own stones          2 stones captured by the move
  3 opponent groups put in atari   4 liberties of the played group  5 self-atari (1 liberty left)
  6 Chebyshev distance to the opponent's previous move (19 if none)
  7 own groups already in atari before the move   8 stones within distance 2 before the move
  9 line (distance to edge)

Output mirrors the feature-shard format (tokens [G, S, 10], masks, game_key, player_id, rank) so
transformer_rank.py can append it with --value-train / --value-val.

Example:
    c1/.venv/bin/python fight_features.py --splits train val holdout --workers 8
"""
import argparse
import pathlib
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numba import njit  # type: ignore[import-untyped]

import board_sim
import rank_lib as lib

sys.path.insert(0, str(lib.ROOT / "c1"))
# Same move parsing / decision sampling as the dense extractor (competition-1 package found
# through the run-time path above, hence after the local imports).
from go_retrieval import katago  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order,import-error

N = 19
K = 10


@njit(cache=True)
def _neighbour(y, x, d):
    """The d-th orthogonal neighbour of (y, x): up, down, left, right."""
    ny = y + (-1 if d == 0 else (1 if d == 1 else 0))
    nx = x + (-1 if d == 2 else (1 if d == 3 else 0))
    return ny, nx


@njit(cache=True)
def _flood(board, y, x, label, mark):
    """Marks the group at (y, x) with `mark` in `label`; returns its liberty count."""
    colour = board[y, x]
    stack = np.empty(N * N, np.int64)
    libs = np.zeros((N, N), np.bool_)
    top = 1
    stack[0] = y * N + x
    label[y, x] = mark
    n_libs = 0
    while top > 0:
        top -= 1
        cy, cx = stack[top] // N, stack[top] % N
        for d in range(4):
            ny, nx = _neighbour(cy, cx, d)
            if ny < 0 or ny >= N or nx < 0 or nx >= N:
                continue
            if board[ny, nx] == 0:
                if not libs[ny, nx]:
                    libs[ny, nx] = True
                    n_libs += 1
            elif board[ny, nx] == colour and label[ny, nx] != mark:
                label[ny, nx] = mark
                stack[top] = ny * N + nx
                top += 1
    return n_libs


@njit(cache=True)
def _adjacent(b0, y, x, me, opp):
    """Adjacent opponent and own stones before the move."""
    adj_opp = 0
    adj_own = 0
    for d in range(4):
        ny, nx = _neighbour(y, x, d)
        if 0 <= ny < N and 0 <= nx < N:
            if b0[ny, nx] == opp:
                adj_opp += 1
            elif b0[ny, nx] == me:
                adj_own += 1
    return adj_opp, adj_own


@njit(cache=True)
def _captures_and_density(b0, b1, y, x, opp):
    """Opponent stones removed by the move and stones within distance 2 before it."""
    opp_before = 0
    opp_after = 0
    density = 0
    for yy in range(N):
        for xx in range(N):
            if b0[yy, xx] == opp:
                opp_before += 1
            if b1[yy, xx] == opp:
                opp_after += 1
            if b0[yy, xx] != 0 and abs(yy - y) <= 2 and abs(xx - x) <= 2:
                density += 1
    return opp_before - opp_after, density


@njit(cache=True)
def _ataris_and_liberties(b1, y, x, me, opp):
    """Opponent groups adjacent to the move that are in atari afterwards (each group once) and
    the liberties of the played group."""
    label = np.zeros((N, N), np.int64)
    ataris = 0
    mark = 1
    for d in range(4):
        ny, nx = _neighbour(y, x, d)
        if 0 <= ny < N and 0 <= nx < N and b1[ny, nx] == opp and label[ny, nx] == 0:
            if _flood(b1, ny, nx, label, mark) == 1:
                ataris += 1
            mark += 1
    own_libs = 0
    if b1[y, x] == me:
        own_libs = _flood(b1, y, x, label, mark)
    return ataris, own_libs


@njit(cache=True)
def _own_groups_in_atari(b0, me):
    """Own groups in atari BEFORE the move (is the player answering an emergency?)."""
    label0 = np.zeros((N, N), np.int64)
    in_atari = 0
    m0 = 1
    for yy in range(N):
        for xx in range(N):
            if b0[yy, xx] == me and label0[yy, xx] == 0:
                if _flood(b0, yy, xx, label0, m0) == 1:
                    in_atari += 1
                m0 += 1
    return in_atari


@njit(cache=True)
def _previous_opponent_distance(pos, mover, t, opp):
    """Chebyshev distance from move t to the opponent's previous non-pass move (19 if none)."""
    y, x = pos[t] // N, pos[t] % N
    dist = 19
    j = t - 1
    while j >= 0:
        if mover[j] == opp - 1 and pos[j] < N * N:
            py, px = pos[j] // N, pos[j] % N
            dist = max(abs(py - y), abs(px - x))
            break
        j -= 1
    return dist


@njit(cache=True)
def decision_features(states, pos, mover, turns, me):
    """states: [n+1, 19, 19] (0 empty, 1 black, 2 white); pos/mover per move;
    me = 1 black or 2 white."""
    opp = 3 - me
    out = np.zeros((len(turns), K), np.float32)
    for i, t in enumerate(turns):
        b0 = states[t]
        b1 = states[t + 1]
        y, x = pos[t] // N, pos[t] % N
        out[i, 0], out[i, 1] = _adjacent(b0, y, x, me, opp)
        out[i, 2], out[i, 8] = _captures_and_density(b0, b1, y, x, opp)
        out[i, 3], own_libs = _ataris_and_liberties(b1, y, x, me, opp)
        out[i, 4] = min(own_libs, 8)
        out[i, 5] = 1.0 if own_libs == 1 else 0.0
        out[i, 6] = _previous_opponent_distance(pos, mover, t, opp)
        out[i, 7] = _own_groups_in_atari(b0, me)
        out[i, 9] = min(x, N - 1 - x, y, N - 1 - y)
    return out


def process_shard(job: tuple[pathlib.Path, int]) -> tuple[str, np.ndarray, np.ndarray,
                                                          pd.DataFrame]:
    """Features of one games_*.csv shard; job = (csv path, decisions per game)."""
    csv_path, samples = job
    games = pd.read_csv(csv_path)
    tokens = np.zeros((len(games), samples, K), np.float32)
    masks = np.zeros((len(games), samples), np.bool_)
    for g, row in enumerate(games.itertuples()):
        moves = katago.parse_sgf_moves(row.sgf_content)
        turns = katago.sample_player_turns(moves, row.color, samples)
        if not turns:
            continue
        pos = np.array([361 if m.x is None else m.y * N + m.x for m in moves], np.int64)
        mover = np.array([0 if m.color == "B" else 1 for m in moves], np.int64)
        states = board_sim.simulate(pos, mover, len(moves))
        tokens[g, :len(turns)] = decision_features(states, pos, mover, np.array(turns, np.int64),
                                                   1 if row.color == "B" else 2)
        masks[g, :len(turns)] = True
    return csv_path.name, tokens, masks, games


def main() -> None:
    """Writes features/fight96/<split>/feat_*.npz for every requested split."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "holdout"])
    ap.add_argument("--samples-per-game", type=int, default=96)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=pathlib.Path, default=lib.ROOT / "features/fight96")
    args = ap.parse_args()
    for split in args.splits:
        shards = sorted((lib.ROOT / f"features/shards/{split}").glob("games_*.csv"))
        out = args.out / split
        out.mkdir(parents=True, exist_ok=True)
        with Pool(args.workers) as pool:
            jobs = [(s, args.samples_per_game) for s in shards]
            for name, tokens, masks, games in pool.imap(process_shard, jobs):
                np.savez(out / name.replace("games_", "feat_").replace(".csv", ".npz"),
                         tokens=tokens, masks=masks,
                         game_key=np.asarray(games["game_key"], dtype="U"),
                         player_id=np.asarray(games["player_id"], dtype="U"),
                         rank=np.asarray(games["rank"].fillna(""), dtype="U"))
                print(split, name, "done", flush=True)


if __name__ == "__main__":
    main()
