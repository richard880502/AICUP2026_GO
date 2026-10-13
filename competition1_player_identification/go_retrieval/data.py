"""Indexed CSV datasets and P-by-K sampling for Go-game retrieval."""

import csv
import collections.abc
import dataclasses
import pathlib
import random
import sqlite3

import torch
import torch.utils.data

from go_retrieval import sgf


@dataclasses.dataclass(frozen=True)
class GameReference:
    """Locates one game within a source CSV file.

    Attributes:
        path: Absolute or working-directory-relative CSV path.
        offset: Byte offset to the CSV record.
        player_index: Consecutive class label used by ArcFace.
    """

    path: str
    offset: int
    player_index: int


@dataclasses.dataclass(frozen=True)
class RetrievalHoldout:
    """Leakage-free candidate and query games reserved for retrieval validation.

    Attributes:
        train_by_player: Game-reference indices eligible for P-by-K training.
        candidate_by_player: Exactly ``candidate_games`` held-out games per
            benchmark player.
        query_by_player: Held-out query games in a stable, seeded order.
    """

    train_by_player: list[list[int]]
    candidate_by_player: dict[int, list[int]]
    query_by_player: dict[int, list[int]]


@dataclasses.dataclass(frozen=True)
class RetrievalHoldoutConfig:
    """Parameters shared by train-time splitting and held-out evaluation.

    Attributes:
        games_per_player: Number of games retained for a train-time group.
        candidate_games: Number of games used to construct each player library.
        query_games: Number of held-out games available for query subsets.
        candidate_players: Number of players in the retrieval pool.
        seed: Random seed for the retrieval pool and each game's split.
        player_color: Optional color restriction applied before splitting.
        excluded_holdout_seed: When set, excludes the retrieval pool selected
            by this seed, creating a player-disjoint validation cohort.
    """

    games_per_player: int
    candidate_games: int
    query_games: int
    candidate_players: int
    seed: int
    player_color: str | None = None
    excluded_holdout_seed: int | None = None


def build_sqlite_index(
    csv_paths: collections.abc.Sequence[pathlib.Path],
    index_path: pathlib.Path,
) -> None:
    """Creates a seekable player-to-game index without loading SGFs into RAM.

    The supplied competition files contain one SGF per physical CSV line. A row
    with an embedded newline is rejected because its byte offset is ambiguous.
    """
    if index_path.exists():
        raise FileExistsError(f"Index already exists at {str(index_path)!r}")
    connection = sqlite3.connect(index_path)
    try:
        connection.executescript(
            """
            CREATE TABLE games (
                player_id TEXT NOT NULL,
                path TEXT NOT NULL,
                offset INTEGER NOT NULL,
                color TEXT NOT NULL
            );
            CREATE INDEX games_by_player_color ON games(player_id, color);
            """
        )
        for csv_path in csv_paths:
            _index_csv_file(connection, csv_path)
        connection.commit()
    finally:
        connection.close()


def _index_csv_file(connection: sqlite3.Connection, csv_path: pathlib.Path) -> None:
    """Indexes one one-line-per-record CSV file."""
    rows: list[tuple[str, str, int, str]] = []
    with csv_path.open("rb") as source_file:
        header = source_file.readline().decode("utf-8")
        columns = next(csv.reader([header]))
        try:
            player_column = columns.index("player_id")
            color_column = columns.index("color")
        except ValueError as error:
            raise ValueError(
                f"CSV has no player_id or color column: {str(csv_path)!r}",
            ) from error
        while raw_row := source_file.readline():
            offset = source_file.tell() - len(raw_row)
            row = next(csv.reader([raw_row.decode("utf-8")]))
            rows.append((row[player_column], str(csv_path), offset, row[color_column]))
            if len(rows) >= 10_000:
                connection.executemany(
                    "INSERT INTO games (player_id, path, offset, color) VALUES (?, ?, ?, ?)",
                    rows,
                )
                rows.clear()
    if rows:
        connection.executemany(
            "INSERT INTO games (player_id, path, offset, color) VALUES (?, ?, ?, ?)",
            rows,
        )


def load_references(
    index_path: pathlib.Path,
    min_games: int,
    player_color: str | None = None,
) -> tuple[list[GameReference], list[list[int]], dict[str, int]]:
    """Loads eligible players and references from an on-disk index.

    Args:
        index_path: SQLite index from :func:`build_sqlite_index`.
        min_games: Drops players having fewer than this many games.

    Returns:
        Flattened game references, their per-player index lists, and the class
        mapping from anonymized player ID to integer label.
    """
    connection = sqlite3.connect(index_path)
    try:
        where_clause, color_arguments = _color_filter(player_color)
        player_ids = _load_eligible_player_ids(
            connection,
            where_clause,
            color_arguments,
            min_games,
        )
        player_to_index = {player_id: index for index, player_id in enumerate(player_ids)}
        references, by_player = _load_game_references(
            connection,
            where_clause,
            color_arguments,
            player_to_index,
        )
    finally:
        connection.close()
    return references, by_player, player_to_index


def _color_filter(player_color: str | None) -> tuple[str, tuple[object, ...]]:
    """Builds an optional SQL color filter and its matching parameters."""
    if player_color is None:
        return "", ()
    return "WHERE color = ?", (player_color,)


def _load_eligible_player_ids(
    connection: sqlite3.Connection,
    where_clause: str,
    color_arguments: tuple[object, ...],
    min_games: int,
) -> list[str]:
    """Returns player IDs having enough games after an optional color filter."""
    rows = connection.execute(
        "SELECT player_id FROM games "
        f"{where_clause} "
        "GROUP BY player_id HAVING COUNT(*) >= ? ORDER BY player_id",
        color_arguments + (min_games,),
    ).fetchall()
    return [row[0] for row in rows]


def _load_game_references(
    connection: sqlite3.Connection,
    where_clause: str,
    color_arguments: tuple[object, ...],
    player_to_index: dict[str, int],
) -> tuple[list[GameReference], list[list[int]]]:
    """Builds sorted references and per-player index lists for eligible IDs."""
    references: list[GameReference] = []
    by_player: list[list[int]] = [[] for _ in player_to_index]
    cursor = connection.execute(
        "SELECT player_id, path, offset FROM games "
        f"{where_clause} "
        "ORDER BY player_id, path, offset",
        color_arguments,
    )
    for player_id, path, offset in cursor:
        player_index = player_to_index.get(player_id)
        if player_index is None:
            continue
        reference_index = len(references)
        references.append(GameReference(path, offset, player_index))
        by_player[player_index].append(reference_index)
    return references, by_player


def make_retrieval_holdout(
    by_player: collections.abc.Sequence[collections.abc.Sequence[int]],
    config: RetrievalHoldoutConfig,
) -> RetrievalHoldout:
    """Reserves candidate and query games without exposing them to training.

    Players are selected from those with enough games to retain a complete
    P-by-K training group after candidate and query games are held out.
    """
    required_games = (
        config.games_per_player + config.candidate_games + config.query_games
    )
    eligible_players = [
        player_index
        for player_index, game_indices in enumerate(by_player)
        if len(game_indices) >= required_games
    ]
    if len(eligible_players) < config.candidate_players:
        raise ValueError(
            f"Only {len(eligible_players)} players have {required_games} games; "
            f"need {config.candidate_players} benchmark players"
        )
    eligible_players = _exclude_holdout_players(eligible_players, config)
    if len(eligible_players) < config.candidate_players:
        raise ValueError(
            "Only "
            f"{len(eligible_players)} players remain after exclusion; need "
            f"{config.candidate_players} benchmark players",
        )
    random_source = random.Random(config.seed)
    selected_players = random_source.sample(eligible_players, config.candidate_players)
    candidate_by_player: dict[int, list[int]] = {}
    query_by_player: dict[int, list[int]] = {}
    train_by_player = [list(game_indices) for game_indices in by_player]
    for player_index in selected_players:
        game_indices = list(by_player[player_index])
        random.Random(config.seed + player_index).shuffle(game_indices)
        candidate_by_player[player_index] = game_indices[:config.candidate_games]
        query_by_player[player_index] = game_indices[
            config.candidate_games:config.candidate_games + config.query_games
        ]
        train_by_player[player_index] = game_indices[
            config.candidate_games + config.query_games:
        ]
    return RetrievalHoldout(train_by_player, candidate_by_player, query_by_player)


def _exclude_holdout_players(
    eligible_players: list[int],
    config: RetrievalHoldoutConfig,
) -> list[int]:
    """Removes a deterministic prior cohort when requested for validation."""
    if config.excluded_holdout_seed is None:
        return eligible_players
    excluded_source = random.Random(config.excluded_holdout_seed)
    excluded_players = set(
        excluded_source.sample(eligible_players, config.candidate_players),
    )
    return [
        player_index
        for player_index in eligible_players
        if player_index not in excluded_players
    ]


def load_retrieval_holdout(
    index_path: pathlib.Path,
    config: RetrievalHoldoutConfig,
) -> tuple[list[GameReference], RetrievalHoldout]:
    """Loads indexed games and applies the deterministic retrieval holdout."""
    references, by_player, _ = load_references(
        index_path,
        config.games_per_player,
        config.player_color,
    )
    return references, make_retrieval_holdout(by_player, config)


class PlayerGameDataset(torch.utils.data.Dataset):
    """Reads tokenized games lazily by indexed byte offset."""

    def __init__(
        self,
        references: collections.abc.Sequence[GameReference],
        max_moves: int,
    ):
        self.references = references
        self.max_moves = max_moves

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        reference = self.references[index]
        row = read_game_row(reference)
        tokens, length = sgf.tokenize_sgf(row["sgf_content"], self.max_moves)
        return torch.from_numpy(tokens), length, reference.player_index


def read_game_row(reference: GameReference) -> dict[str, str]:
    """Reads one source CSV row identified by its indexed byte offset."""
    with open(reference.path, "rb") as source_file:
        source_file.seek(reference.offset)
        row = next(csv.DictReader(
            [source_file.readline().decode("utf-8")],
            fieldnames=("player_id", "game_id", "rank", "color", "sgf_content"),
        ))
    return {key: value for key, value in row.items() if key is not None and value is not None}


class PKBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Samples P players and K distinct games per player in every batch."""

    def __init__(
        self,
    by_player: collections.abc.Sequence[collections.abc.Sequence[int]],
        players_per_batch: int,
        games_per_player: int,
        batches_per_epoch: int,
        seed: int,
    ):
        super().__init__()
        if players_per_batch > len(by_player):
            raise ValueError("players_per_batch exceeds eligible player count")
        self.by_player = by_player
        self.players_per_batch = players_per_batch
        self.games_per_player = games_per_player
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0
        if any(len(games) < games_per_player for games in by_player):
            raise ValueError("Every sampled player needs at least games_per_player games")

    def set_epoch(self, epoch: int) -> None:
        """Sets a deterministic epoch-dependent random seed."""
        self.epoch = epoch

    def __iter__(self) -> collections.abc.Iterator[list[int]]:
        random_source = random.Random(self.seed + self.epoch)
        player_indices = range(len(self.by_player))
        for _ in range(self.batches_per_epoch):
            batch: list[int] = []
            for player_index in random_source.sample(
                player_indices,
                self.players_per_batch,
            ):
                batch.extend(random_source.sample(
                    self.by_player[player_index],
                    self.games_per_player,
                ))
            yield batch

    def __len__(self) -> int:
        return self.batches_per_epoch


def collate_games(
    items: collections.abc.Sequence[tuple[torch.Tensor, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacks game tokens, lengths, and player labels into a training batch."""
    tokens, lengths, labels = zip(*items, strict=True)
    return (
        torch.stack(tokens),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
