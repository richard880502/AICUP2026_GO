"""Extracts resumable KataGo features for an official test pool.

The command reads one candidate library and its matching test questions. It
writes small immutable shards so a long production extraction can resume after
an interruption without recomputing completed games.

Example:
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python extract_submission_features.py \
        --candidates tests/player_identification_candidates_public.csv \
        --questions tests/player_identification_test_public.csv \
        --output production/20261030/public/features \
        --engine scripts/run_katago.sh \
        --engine-config configs/analysis_1x3090.cfg \
        --model models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz \
        --human-model models/katago/b18c384nbt-humanv0.bin.gz
"""

import argparse
import collections.abc
import csv
import dataclasses
import json
import pathlib
import time

import numpy as np

from go_retrieval import katago


DEFAULT_HUMAN_PROFILES = ("rank_5k", "rank_1d", "rank_5d")


@dataclasses.dataclass(frozen=True)
class GameRecord:
    """Stores one game and the output group it belongs to."""

    record_id: str
    group_index: int
    sgf_content: str
    color: str


@dataclasses.dataclass(frozen=True)
class ExtractionConfig:
    """Controls feature extraction and shard granularity."""

    samples_per_game: int
    max_visits: int
    engine_batch_size: int
    shard_size: int
    human_profiles: tuple[str, ...]
    worker_index: int = 0
    num_workers: int = 1


@dataclasses.dataclass(frozen=True)
class PreparedRecord:
    """Stores parsed moves and request metadata for one game."""

    record: GameRecord
    request_id: str
    moves: list[katago.SgfMove]
    turns: list[int]


@dataclasses.dataclass(frozen=True)
class ExtractionGroup:
    """Names one candidate or query collection and its destination."""

    name: str
    records: list[GameRecord]
    output_root: pathlib.Path


@dataclasses.dataclass(frozen=True)
class MetadataContext:
    """Stores stable identifiers written beside production shards."""

    player_ids: list[str]
    question_ids: list[str]
    game_counts: list[int]


def parse_arguments() -> argparse.Namespace:
    """Parses official CSV, KataGo, and resumable-output settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=pathlib.Path, required=True)
    parser.add_argument("--questions", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--engine", type=pathlib.Path, required=True)
    parser.add_argument("--engine-config", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--human-model", type=pathlib.Path, required=True)
    parser.add_argument(
        "--human-profiles",
        nargs="+",
        default=DEFAULT_HUMAN_PROFILES,
    )
    parser.add_argument("--samples-per-game", type=int, default=64)
    parser.add_argument("--max-visits", type=int, default=1)
    parser.add_argument("--engine-batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def _read_candidates(
    path: pathlib.Path,
) -> tuple[list[GameRecord], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    player_ids = sorted({row["player_id"] for row in rows})
    player_indices = {
        player_id: index
        for index, player_id in enumerate(player_ids)
    }
    records = []
    for row in rows:
        records.append(GameRecord(
            row["game_id"],
            player_indices[row["player_id"]],
            row["sgf_content"],
            row["color"],
        ))
    return records, player_ids


def _read_questions(
    path: pathlib.Path,
) -> tuple[list[GameRecord], list[str], list[int]]:
    records = []
    question_ids = []
    game_counts = []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        for question_index, row in enumerate(csv.DictReader(input_file)):
            question_id = row["question_id"]
            game_count = int(row["num_games"])
            question_ids.append(question_id)
            game_counts.append(game_count)
            for game_offset in range(1, game_count + 1):
                records.append(GameRecord(
                    f"{question_id}:{game_offset}",
                    question_index,
                    row[f"sgf_{game_offset}"],
                    row[f"color_{game_offset}"],
                ))
    return records, question_ids, game_counts


def _chunks(
    records: collections.abc.Sequence[GameRecord],
    chunk_size: int,
) -> collections.abc.Iterator[collections.abc.Sequence[GameRecord]]:
    for start in range(0, len(records), chunk_size):
        yield records[start:start + chunk_size]


def _queries_for_record(
    record: GameRecord,
    request_id: str,
    config: ExtractionConfig,
) -> tuple[
    list[katago.SgfMove],
    list[int],
    list[dict[str, object]],
]:
    moves = katago.parse_sgf_moves(record.sgf_content)
    turns = katago.sample_player_turns(
        moves,
        record.color,
        config.samples_per_game,
    )
    queries = [katago.make_analysis_query(
        request_id,
        moves,
        turns,
        config.max_visits,
        katago.AnalysisQueryConfig(include_transition_features=True),
    )]
    for profile_index, profile in enumerate(config.human_profiles):
        queries.append(katago.make_analysis_query(
            f"{request_id}:human:{profile_index}",
            moves,
            turns,
            config.max_visits,
            katago.AnalysisQueryConfig(
                include_transition_features=True,
                human_sl_profile=profile,
            ),
        ))
    return moves, turns, queries


def _extract_batch(
    client: katago.KataGoClient,
    records: collections.abc.Sequence[GameRecord],
    config: ExtractionConfig,
    request_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    prepared_records, queries = _prepare_batch(
        records, config, request_prefix,
    )
    responses = client.analyze(queries)
    token_rows = []
    for prepared in prepared_records:
        token_rows.append(_tokens_from_responses(
            prepared, responses, config.human_profiles,
        ))
    feature_count = len(katago.decision_feature_names(
        True, config.human_profiles,
    ))
    tokens = np.zeros(
        (len(records), config.samples_per_game, feature_count),
        dtype=np.float32,
    )
    masks = np.zeros(
        (len(records), config.samples_per_game), dtype=np.bool_,
    )
    for record_index, decision_tokens in enumerate(token_rows):
        tokens[record_index, :len(decision_tokens)] = decision_tokens
        masks[record_index, :len(decision_tokens)] = True
    return tokens, masks


def _prepare_batch(
    records: collections.abc.Sequence[GameRecord],
    config: ExtractionConfig,
    request_prefix: str,
) -> tuple[list[PreparedRecord], list[dict[str, object]]]:
    prepared_records = []
    queries = []
    for record_index, record in enumerate(records):
        request_id = f"{request_prefix}:{record_index}"
        moves, turns, record_queries = _queries_for_record(
            record,
            request_id,
            config,
        )
        prepared_records.append(PreparedRecord(
            record, request_id, moves, turns,
        ))
        queries.extend(record_queries)
    return prepared_records, queries


def _tokens_from_responses(
    prepared: PreparedRecord,
    responses: dict[str, list[dict[str, object]]],
    human_profiles: tuple[str, ...],
) -> np.ndarray:
    human_responses = {}
    for profile_index, profile in enumerate(human_profiles):
        response_id = f"{prepared.request_id}:human:{profile_index}"
        human_responses[profile] = responses[response_id]
    return katago.build_decision_tokens(
        prepared.moves,
        prepared.record.color,
        prepared.turns,
        katago.DecisionResponses(
            responses[prepared.request_id], human_responses,
        ),
        include_transition_features=True,
    )


def _atomic_save(
    path: pathlib.Path,
    arrays: dict[str, np.ndarray],
) -> None:
    temporary_path = path.with_suffix(".tmp")
    with temporary_path.open("wb") as output_file:
        # NumPy's overload incorrectly treats arbitrary named arrays as bools.
        np.savez_compressed(output_file, **arrays)  # type: ignore[arg-type]
    temporary_path.replace(path)


def _extract_shard(
    client: katago.KataGoClient,
    records: collections.abc.Sequence[GameRecord],
    config: ExtractionConfig,
    request_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    token_batches = []
    mask_batches = []
    for batch_index, batch in enumerate(
        _chunks(records, config.engine_batch_size),
    ):
        tokens, masks = _extract_batch(
            client,
            batch,
            config,
            f"{request_prefix}:{batch_index}",
        )
        token_batches.append(tokens)
        mask_batches.append(masks)
    return np.concatenate(token_batches), np.concatenate(mask_batches)


def _write_metadata(
    output_root: pathlib.Path,
    arguments: argparse.Namespace,
    context: MetadataContext,
    config: ExtractionConfig,
) -> None:
    metadata = {
        "candidates": str(arguments.candidates.resolve()),
        "questions": str(arguments.questions.resolve()),
        "player_ids": context.player_ids,
        "question_ids": context.question_ids,
        "query_game_counts": context.game_counts,
        "samples_per_game": config.samples_per_game,
        "max_visits": config.max_visits,
        "human_profiles": list(config.human_profiles),
        "features": list(katago.decision_feature_names(
            True,
            config.human_profiles,
        )),
        "model": str(arguments.model.resolve()),
        "human_model": str(arguments.human_model.resolve()),
    }
    metadata_path = output_root / "metadata.json"
    serialized = json.dumps(metadata, indent=2) + "\n"
    if metadata_path.exists():
        if metadata_path.read_text(encoding="utf-8") != serialized:
            raise ValueError(
                f"Existing metadata does not match this run: {metadata_path}",
            )
        return
    metadata_path.write_text(serialized, encoding="utf-8")


def _extract_group(
    client: katago.KataGoClient,
    group: ExtractionGroup,
    config: ExtractionConfig,
) -> None:
    group_root = group.output_root / group.name
    group_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = 0
    processed = 0
    total = sum(
        len(group.records[start:start + config.shard_size])
        for start in range(config.worker_index * config.shard_size,
                           len(group.records),
                           config.num_workers * config.shard_size)
    )
    for shard_index, shard_records in enumerate(
        _chunks(group.records, config.shard_size),
    ):
        if shard_index % config.num_workers != config.worker_index:
            continue
        shard_path = group_root / f"shard_{shard_index:05d}.npz"
        if shard_path.exists():
            completed += len(shard_records)
            continue
        tokens, masks = _extract_shard(
            client,
            shard_records,
            config,
            f"{group.name}:{shard_index}",
        )
        _atomic_save(
            shard_path,
            {
                "tokens": tokens,
                "masks": masks,
                "group_indices": np.asarray(
                    [record.group_index for record in shard_records],
                    dtype=np.int32,
                ),
                "record_ids": np.asarray(
                    [record.record_id for record in shard_records],
                ),
            },
        )
        completed += len(shard_records)
        processed += len(shard_records)
        rate = processed / max(time.monotonic() - started, 1e-6)
        remaining_seconds = (total - completed) / max(rate, 1e-6)
        print(
            f"{group.name}: {completed}/{total} games "
            f"({rate:.2f} games/s, ETA {remaining_seconds / 3600:.2f}h)",
            flush=True,
        )


def _validate_arguments(arguments: argparse.Namespace) -> None:
    paths = (
        arguments.candidates,
        arguments.questions,
        arguments.engine,
        arguments.engine_config,
        arguments.model,
        arguments.human_model,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    numeric_values = (
        arguments.samples_per_game,
        arguments.max_visits,
        arguments.engine_batch_size,
        arguments.shard_size,
        arguments.num_workers,
    )
    if any(value <= 0 for value in numeric_values):
        raise ValueError("Extraction sizes and max-visits must be positive")
    if not 0 <= arguments.worker_index < arguments.num_workers:
        raise ValueError("worker-index must be between zero and num-workers - 1")


def main() -> None:
    """Extracts candidate and query features into resumable shards."""
    arguments = parse_arguments()
    _validate_arguments(arguments)
    config = ExtractionConfig(
        arguments.samples_per_game,
        arguments.max_visits,
        arguments.engine_batch_size,
        arguments.shard_size,
        tuple(arguments.human_profiles),
        arguments.worker_index,
        arguments.num_workers,
    )
    candidates, player_ids = _read_candidates(arguments.candidates)
    queries, question_ids, game_counts = _read_questions(
        arguments.questions,
    )
    if len(player_ids) < 5:
        raise ValueError(f"Top-5 retrieval requires at least five players: {len(player_ids)}")
    arguments.output.mkdir(parents=True, exist_ok=True)
    _write_metadata(
        arguments.output,
        arguments,
        MetadataContext(player_ids, question_ids, game_counts),
        config,
    )
    with katago.KataGoClient(
        arguments.engine,
        arguments.engine_config,
        arguments.model,
        arguments.human_model,
    ) as client:
        _extract_group(
            client,
            ExtractionGroup(
                "candidates", candidates, arguments.output,
            ),
            config,
        )
        _extract_group(
            client,
            ExtractionGroup("queries", queries, arguments.output),
            config,
        )


if __name__ == "__main__":
    main()
