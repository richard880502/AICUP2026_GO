# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Audit whether exact duplicated games reconstruct a useful matchmaking graph.

This is the P0 gate for identity-conditioned matchmaking residuals. It does
not train a neural model and does not read official labels.

The rank training cache stores a canonical SHA256 digest of setup + move
sequence. If the same physical game appears once for each labelled player,
the two rows have the same digest but different player_id/color. Those pairs
give an exact player-player graph with no opponent-identity inference.

The audit reports:
1. how many canonical games yield exact two-player edges;
2. graph coverage/degree over all labelled players and the 6,807 v47 classes;
3. optional official v4-top1 coverage of graph-covered classes;
4. whether same-rank player identities have a stable matchmaking-neighbour
   residual under a deterministic split of DISTINCT neighbours.

Run from repository root:
    PYTHONPATH=.:competition1_player_identification \
      competition2_rank_prediction/c1/.venv/bin/python -m \
      competition2_rank_prediction.experiments.opponent_identity.\
audit_identity_matchmaking_graph
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
from collections import defaultdict

import numpy as np
from scipy.stats import pearsonr, spearmanr  # type: ignore[import-untyped]

from competition1_player_identification.experiments.open_set_v2 import cache
from competition2_rank_prediction.experiments.opponent_identity import link_inputs

RANKS = tuple(cache.RANKS)


def as_bytes(values: np.ndarray, width: int) -> np.ndarray:
    """Stores string arrays compactly for the million-row audit."""
    return np.asarray(values).astype(f"S{width}", copy=False)


def load_rows(root: pathlib.Path) -> dict:
    """Loads only fields needed for exact-game grouping."""
    hashes, players, ranks, colors, encoded = [], [], [], [], []
    player_shards: list[np.ndarray] = []
    total = 0
    for rank_index, rank in enumerate(RANKS):
        with np.load(root / f"{rank}.npz") as shard:
            h = as_bytes(np.asarray(shard["hashes"]), 64)
            p = as_bytes(np.asarray(shard["players"]), 128)
            c = np.asarray(shard["colors"], dtype=np.int8)
        # Positions run 0..len-1, so exactly 100000 rows still fit in the block
        # this rank owns; the same convention is used by h3_dump.py.
        if len(h) > 100000:
            raise ValueError(
                f"{rank} has {len(h)} rows; encoded row convention assumes <=100000"
            )
        hashes.append(h)
        players.append(p)
        ranks.append(np.full(len(h), rank_index, dtype=np.int8))
        colors.append(c)
        encoded.append(
            rank_index * 100000 + np.arange(len(h), dtype=np.int64)
        )
        player_shards.append(p)
        total += len(h)
        print(f"loaded {rank}: {len(h):,} rows ({total:,} total)", flush=True)
    return {
        "hashes": np.concatenate(hashes),
        "players": np.concatenate(players),
        "ranks": np.concatenate(ranks),
        "colors": np.concatenate(colors),
        "encoded": np.concatenate(encoded),
        "player_shards": player_shards,
    }


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def exact_edges(rows: dict) -> tuple[list[dict], dict]:
    """Groups canonical hashes and returns exact two-player/opposite-color edges."""
    hashes = rows["hashes"]
    order = np.argsort(hashes, kind="stable")
    sorted_hashes = hashes[order]
    boundaries = np.flatnonzero(sorted_hashes[1:] != sorted_hashes[:-1]) + 1
    groups = np.split(order, boundaries)

    edges: list[dict] = []
    duplicated = same_player_only = multi_player = two_player = 0
    opposite_color = same_color = 0
    duplicate_rows = 0
    for group in groups:
        if len(group) < 2:
            continue
        duplicated += 1
        duplicate_rows += len(group)
        members: dict[bytes, list[int]] = defaultdict(list)
        for row in group:
            members[bytes(rows["players"][row])].append(int(row))
        if len(members) == 1:
            same_player_only += 1
            continue
        if len(members) != 2:
            multi_player += 1
            continue
        two_player += 1
        items = sorted(members.items(), key=lambda item: item[0])
        (player_a, rows_a), (player_b, rows_b) = items
        rank_a_values = np.unique(rows["ranks"][rows_a])
        rank_b_values = np.unique(rows["ranks"][rows_b])
        if len(rank_a_values) != 1 or len(rank_b_values) != 1:
            raise ValueError("one player appears under multiple ranks in one game")
        color_a_values = np.unique(rows["colors"][rows_a])
        color_b_values = np.unique(rows["colors"][rows_b])
        is_opposite = (
            len(color_a_values) == 1
            and len(color_b_values) == 1
            and int(color_a_values[0]) != int(color_b_values[0])
        )
        if is_opposite:
            opposite_color += 1
        else:
            same_color += 1
        # Keep only physically plausible exact pairs in the graph.
        if not is_opposite:
            continue
        edges.append(
            {
                "hash": bytes(rows["hashes"][group[0]]).decode(),
                "player_a": player_a.decode(),
                "rank_a": int(rank_a_values[0]),
                "color_a": int(color_a_values[0]),
                "player_b": player_b.decode(),
                "rank_b": int(rank_b_values[0]),
                "color_b": int(color_b_values[0]),
                "rows_a": len(rows_a),
                "rows_b": len(rows_b),
            }
        )
    report = {
        "rows": int(len(hashes)),
        "unique_canonical_games": int(len(groups)),
        "duplicate_hash_groups": duplicated,
        "rows_inside_duplicate_groups": duplicate_rows,
        "same_player_only_duplicate_groups": same_player_only,
        "exactly_two_distinct_player_groups": two_player,
        "more_than_two_player_groups": multi_player,
        "two_player_opposite_color_groups": opposite_color,
        "two_player_same_or_ambiguous_color_groups": same_color,
        "accepted_exact_edges": len(edges),
    }
    return edges, report


def graph_degrees(edges: list[dict]) -> tuple[dict[str, set[str]], dict[tuple[str, str], int]]:
    """Unique-neighbour graph and number of exact games per player pair."""
    neighbours: dict[str, set[str]] = defaultdict(set)
    pair_games: dict[tuple[str, str], int] = defaultdict(int)
    for edge in edges:
        left, right = edge["player_a"], edge["player_b"]
        neighbours[left].add(right)
        neighbours[right].add(left)
        pair_games[tuple(sorted((left, right)))] += 1
    return neighbours, pair_games


def degree_summary(players: list[str], neighbours: dict[str, set[str]]) -> dict:
    """Coverage and degree distribution for a fixed player population."""
    degree = np.asarray([len(neighbours.get(player, set())) for player in players])
    covered = degree > 0
    if not degree.size:
        return {"players": 0}
    return {
        "players": int(len(degree)),
        "covered": int(covered.sum()),
        "coverage": float(covered.mean()),
        "degree_mean": float(degree.mean()),
        "degree_median_all": float(np.median(degree)),
        "degree_median_covered": (
            float(np.median(degree[covered])) if covered.any() else 0.0
        ),
        "degree_p90": float(np.percentile(degree, 90)),
        "degree_max": int(degree.max()),
        "degree_ge_2": int((degree >= 2).sum()),
        "degree_ge_3": int((degree >= 3).sum()),
        "degree_ge_5": int((degree >= 5).sum()),
        "degree_ge_10": int((degree >= 10).sum()),
    }


def class_players(rows: dict) -> list[str]:
    """Maps each v47 class index to its labelled player id."""
    owners, class_rows, selected = link_inputs.v4_gallery()
    count = int(np.asarray(owners).max()) + 1
    output: list[str | None] = [None] * count
    for owner, position in zip(np.asarray(owners), np.asarray(class_rows)):
        key = int(selected[int(position)])
        rank_index, row = divmod(key, 100000)
        if not 0 <= rank_index < len(RANKS):
            raise ValueError(f"invalid encoded gallery row {key}")
        shard = rows["player_shards"][rank_index]
        if not 0 <= row < len(shard):
            raise ValueError(f"gallery row {key} outside raw cache")
        player = bytes(shard[row]).decode()
        previous = output[int(owner)]
        if previous is not None and previous != player:
            raise ValueError(f"class {owner} maps to two players")
        output[int(owner)] = player
    if any(player is None for player in output):
        missing = sum(player is None for player in output)
        raise ValueError(f"{missing} v47 classes have no player mapping")
    return [str(player) for player in output]


def deterministic_half(player: str, neighbour: str) -> int:
    """Stable per-player split of DISTINCT neighbours."""
    digest = hashlib.sha256(f"{player}\0{neighbour}".encode()).digest()
    return digest[0] & 1


def rank_of_players(rows: dict) -> dict[str, int]:
    """Registered rank for every labelled player; validates consistency."""
    result: dict[str, int] = {}
    for player_b, rank in zip(rows["players"], rows["ranks"]):
        player = bytes(player_b).decode()
        rank_int = int(rank)
        previous = result.setdefault(player, rank_int)
        if previous != rank_int:
            raise ValueError(f"player {player} occurs under multiple ranks")
    return result


# Archived research script: one linear procedure, intentionally not split.
# pylint: disable-next=too-many-locals
def residual_audit(
    neighbours: dict[str, set[str]],
    player_rank: dict[str, int],
) -> dict:
    """Split-half stability of per-identity matchmaking-neighbour offsets.

    For player j, the raw signal is the mean registered rank of DISTINCT
    neighbours. We subtract the mean for all players with the same registered
    rank, so delta_j is identity-specific rather than a restatement of rank_j.
    Neighbours are deterministically split into two sets and deltas are
    recomputed independently.
    """
    players = sorted(player_rank)
    rank_baseline = np.zeros(len(RANKS), dtype=np.float64)
    rank_counts = np.zeros(len(RANKS), dtype=np.int64)
    full_mean: dict[str, float] = {}
    for player in players:
        ranks = [player_rank[n] for n in neighbours.get(player, set()) if n in player_rank]
        if not ranks:
            continue
        value = float(np.mean(ranks))
        full_mean[player] = value
        rank = player_rank[player]
        rank_baseline[rank] += value
        rank_counts[rank] += 1
    for rank in range(len(RANKS)):
        if rank_counts[rank]:
            rank_baseline[rank] /= rank_counts[rank]
        else:
            rank_baseline[rank] = float(rank)

    half_delta: dict[str, tuple[float, float, int, int]] = {}
    for player in players:
        left: list[int] = []
        right: list[int] = []
        for neighbour in neighbours.get(player, set()):
            if neighbour not in player_rank:
                continue
            target = left if deterministic_half(player, neighbour) == 0 else right
            target.append(player_rank[neighbour])
        if left and right:
            baseline = rank_baseline[player_rank[player]]
            half_delta[player] = (
                float(np.mean(left) - baseline),
                float(np.mean(right) - baseline),
                len(left),
                len(right),
            )

    thresholds: dict[str, dict] = {}
    for minimum_half in (1, 2, 3, 5):
        selected = [
            value for value in half_delta.values()
            if value[2] >= minimum_half and value[3] >= minimum_half
        ]
        if len(selected) < 3:
            thresholds[str(minimum_half)] = {
                "players": len(selected),
                "pearson": None,
                "spearman": None,
            }
            continue
        first = np.asarray([value[0] for value in selected])
        second = np.asarray([value[1] for value in selected])
        thresholds[str(minimum_half)] = {
            "players": len(selected),
            "pearson": float(pearsonr(first, second).statistic),
            "spearman": float(spearmanr(first, second).statistic),
            "delta_std_half0": float(first.std()),
            "delta_std_half1": float(second.std()),
        }

    return {
        "definition": (
            "delta_j = mean(distinct-neighbour registered rank) - "
            "mean neighbour rank among players sharing j's registered rank"
        ),
        "rank_baseline_mean_neighbour_rank": {
            RANKS[index]: float(rank_baseline[index])
            for index in range(len(RANKS))
        },
        "players_with_two_nonempty_halves": len(half_delta),
        "split_half_by_minimum_neighbours_per_half": thresholds,
    }


def official_top1_coverage(
    scores_path: pathlib.Path | None,
    controls_path: pathlib.Path | None,
    class_player: list[str],
    neighbours: dict[str, set[str]],
) -> dict | None:
    """Cheap official sanity check using cached v4 top-1 classes only."""
    if scores_path is None or not scores_path.exists():
        return None
    with np.load(scores_path) as saved:
        top_class = np.asarray(saved["index"], dtype=np.int64)
        top_value = np.asarray(saved["top"], dtype=np.float64)
    covered_class = np.asarray(
        [len(neighbours.get(player, set())) > 0 for player in class_player],
        dtype=bool,
    )
    result = {
        "note": (
            "This is v4 top-1 coverage, not full v4+v7 posterior mass. It is "
            "only a cheap P0 sanity check."
        ),
        "games": int(len(top_class)),
        "top1_graph_covered": int(covered_class[top_class].sum()),
        "top1_graph_coverage": float(covered_class[top_class].mean()),
    }
    if controls_path is not None and controls_path.exists():
        with np.load(controls_path) as controls:
            threshold = float(np.percentile(controls["a_top"], 99.0))
        confident = top_value > threshold
        result.update(
            {
                "absent_q99_threshold": threshold,
                "confident_games": int(confident.sum()),
                "confident_top1_graph_coverage": (
                    float(covered_class[top_class[confident]].mean())
                    if confident.any() else None
                ),
            }
        )
    return result


def write_edges(path: pathlib.Path, edges: list[dict]) -> None:
    """Writes accepted exact physical-game edges."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "hash", "player_a", "rank_a", "color_a",
        "player_b", "rank_b", "color_b", "rows_a", "rows_b",
    )
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edges)


def main() -> None:
    """Runs the exact-graph and split-half identity-residual audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cache",
        type=pathlib.Path,
        default=pathlib.Path("production/open_set_v2_20261111/raw_cache"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path(
            "runtime/competition2_research/experiments/"
            "identity_matchmaking_graph"
        ),
    )
    parser.add_argument(
        "--official-scores",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/opp_probe/test_opp_scores.npz"
        ),
    )
    parser.add_argument(
        "--controls",
        type=pathlib.Path,
        default=pathlib.Path(
            "/tmp2/richard/claude_scratch/opp_probe/controls.npz"
        ),
    )
    args = parser.parse_args()

    rows = load_rows(args.raw_cache)
    edges, exact_report = exact_edges(rows)
    neighbours, pair_games = graph_degrees(edges)
    player_rank = rank_of_players(rows)

    all_players = sorted(player_rank)
    classes = class_players(rows)
    report = {
        "exact_game_pairing": exact_report,
        "graph": {
            "unique_player_pairs": len(pair_games),
            "pairs_with_2plus_exact_games": int(
                sum(value >= 2 for value in pair_games.values())
            ),
            "pairs_with_5plus_exact_games": int(
                sum(value >= 5 for value in pair_games.values())
            ),
            "all_labelled_players": degree_summary(all_players, neighbours),
            "v47_class_players": degree_summary(classes, neighbours),
        },
        "identity_residual": residual_audit(neighbours, player_rank),
        "official_v4_top1": official_top1_coverage(
            args.official_scores,
            args.controls,
            classes,
            neighbours,
        ),
    }

    args.output.mkdir(parents=True, exist_ok=True)
    write_edges(args.output / "exact_edges.csv", edges)
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
