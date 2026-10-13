"""Caches KataGo decision tokens for a leakage-free retrieval split.

Example:
    .venv/bin/python extract_katago_features.py \
      --output experiments/10_katago_decision/features/holdout_400x100 \
      --samples-per-game 8 --max-visits 1
"""

import argparse
import collections.abc
import dataclasses
import json
import pathlib
import random

import numpy as np

from go_retrieval import data
from go_retrieval import katago


@dataclasses.dataclass(frozen=True)
class FeatureTask:
    """One held-out game and the player identity it belongs to."""

    reference_index: int
    player_index: int
    group: str


@dataclasses.dataclass(frozen=True)
class ExtractionConfig:
    """KataGo sampling and batching settings for a feature cache."""

    samples_per_game: int
    max_visits: int
    games_per_engine_batch: int
    include_transition_features: bool
    human_sl_profiles: tuple[str, ...] = ()
    human_features_only: bool = False


@dataclasses.dataclass(frozen=True)
class PreparedTask:
    """Parsed data required to turn an engine response into feature tokens."""

    task: FeatureTask
    moves: list[katago.SgfMove]
    turns: list[int]
    player_color: str
    request_id: str


@dataclasses.dataclass
class FeatureArrays:
    """Padded feature-cache arrays populated incrementally by engine batches."""

    tokens: np.ndarray
    masks: np.ndarray
    player_indices: np.ndarray


@dataclasses.dataclass
class FeatureExtractor:
    """Owns state shared by every engine batch in one cache extraction."""

    client: katago.KataGoClient
    references: list[data.GameReference]
    extraction_config: ExtractionConfig
    feature_arrays: FeatureArrays

    def write_batch(
        self,
        tasks: collections.abc.Sequence[FeatureTask],
        start_index: int,
    ) -> None:
        """Runs one engine batch and writes its tokens into allocated arrays."""
        prepared_tasks, queries = prepare_batch(
            self.references,
            tasks,
            start_index,
            self.extraction_config,
        )
        responses = self.client.analyze(queries)
        for batch_offset, prepared_task in enumerate(prepared_tasks):
            primary_responses, human_responses = select_task_responses(
                responses,
                prepared_task.request_id,
                self.extraction_config.human_sl_profiles,
                self.extraction_config.human_features_only,
            )
            tokens = katago.build_decision_tokens(
                prepared_task.moves,
                prepared_task.player_color,
                prepared_task.turns,
                katago.DecisionResponses(primary_responses, human_responses or {}),
                self.extraction_config.include_transition_features,
            )
            result_index = start_index + batch_offset
            self.feature_arrays.tokens[result_index, :len(tokens)] = tokens
            self.feature_arrays.masks[result_index, :len(tokens)] = True
            self.feature_arrays.player_indices[result_index] = prepared_task.task.player_index


def parse_arguments() -> argparse.Namespace:
    """Parses engine, split, and feature-cache configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=pathlib.Path, default="training/games.sqlite")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--engine", type=pathlib.Path, default="scripts/run_katago.sh")
    parser.add_argument(
        "--engine-config",
        type=pathlib.Path,
        default="experiments/10_katago_decision/configs/analysis_2x3090.cfg",
    )
    parser.add_argument(
        "--human-model",
        type=pathlib.Path,
        help="KataGo HumanSL model used to emit human-policy features.",
    )
    parser.add_argument(
        "--human-sl-profiles",
        nargs="+",
        default=(),
        help="HumanSL profiles, such as rank_1d rank_5d.",
    )
    parser.add_argument(
        "--human-features-only",
        action="store_true",
        help="Skips normal-model queries when only HumanSL columns are needed.",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default="models/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
    )
    parser.add_argument("--candidate-players", type=int, default=400)
    parser.add_argument("--candidate-games", type=int, default=100)
    parser.add_argument("--query-games", type=int, default=20)
    parser.add_argument("--games-per-player", type=int, default=8)
    parser.add_argument("--holdout-seed", type=int, default=20260920)
    parser.add_argument(
        "--exclude-holdout-seed",
        type=int,
        help="Excludes the player pool selected by this seed.",
    )
    parser.add_argument("--player-color", choices=("B", "W"))
    parser.add_argument("--split", choices=("holdout", "train"), default="holdout")
    parser.add_argument("--train-games-per-player", type=int, default=8)
    parser.add_argument("--samples-per-game", type=int, default=8)
    parser.add_argument("--max-visits", type=int, default=1)
    parser.add_argument("--games-per-engine-batch", type=int, default=4)
    parser.add_argument("--include-transition-features", action="store_true")
    return parser.parse_args()


def make_tasks(
    holdout: data.RetrievalHoldout,
) -> tuple[list[FeatureTask], list[FeatureTask]]:
    """Creates stable candidate and query task lists from a holdout split."""
    candidate_tasks: list[FeatureTask] = []
    query_tasks: list[FeatureTask] = []
    for player_index in sorted(holdout.candidate_by_player):
        candidate_tasks.extend(
            FeatureTask(reference_index, player_index, "candidate")
            for reference_index in holdout.candidate_by_player[player_index]
        )
        query_tasks.extend(
            FeatureTask(reference_index, player_index, "query")
            for reference_index in holdout.query_by_player[player_index]
        )
    return candidate_tasks, query_tasks


def make_train_tasks(
    holdout: data.RetrievalHoldout,
    games_per_player: int,
    seed: int,
) -> list[FeatureTask]:
    """Selects disjoint, class-balanced training games from the train split."""
    if games_per_player <= 0:
        raise ValueError("train_games_per_player must be positive")
    tasks: list[FeatureTask] = []
    for player_index, game_indices in enumerate(holdout.train_by_player):
        if len(game_indices) < games_per_player:
            raise ValueError(
                f"Player {player_index} has only {len(game_indices)} training games; "
                f"need {games_per_player}",
            )
        selected_indices = list(game_indices)
        random.Random(seed + player_index).shuffle(selected_indices)
        tasks.extend(
            FeatureTask(reference_index, player_index, "train")
            for reference_index in selected_indices[:games_per_player]
        )
    return tasks


def chunks(
    items: collections.abc.Sequence[FeatureTask],
    chunk_size: int,
) -> collections.abc.Iterator[collections.abc.Sequence[FeatureTask]]:
    """Yields fixed-size batches without making a second in-memory copy."""
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def extract_tasks(
    client: katago.KataGoClient,
    references: list[data.GameReference],
    tasks: collections.abc.Sequence[FeatureTask],
    extraction_config: ExtractionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns padded decision tokens, masks, and labels for held-out games."""
    feature_arrays = initialize_feature_arrays(tasks, extraction_config)
    extractor = FeatureExtractor(client, references, extraction_config, feature_arrays)
    for task_offset, task_batch in enumerate(
        chunks(tasks, extraction_config.games_per_engine_batch),
    ):
        extractor.write_batch(
            task_batch,
            task_offset * extraction_config.games_per_engine_batch,
        )
    return (
        feature_arrays.tokens,
        feature_arrays.masks,
        feature_arrays.player_indices,
    )


def initialize_feature_arrays(
    tasks: collections.abc.Sequence[FeatureTask],
    extraction_config: ExtractionConfig,
) -> FeatureArrays:
    """Allocates the fixed-shape cache arrays before engine processing begins."""
    feature_count = len(
        katago.decision_feature_names(
            extraction_config.include_transition_features,
            extraction_config.human_sl_profiles,
        ),
    )
    return FeatureArrays(
        np.zeros(
            (len(tasks), extraction_config.samples_per_game, feature_count),
            dtype=np.float32,
        ),
        np.zeros(
            (len(tasks), extraction_config.samples_per_game),
            dtype=np.bool_,
        ),
        np.empty(len(tasks), dtype=np.int32),
    )


def prepare_batch(
    references: list[data.GameReference],
    tasks: collections.abc.Sequence[FeatureTask],
    start_index: int,
    extraction_config: ExtractionConfig,
) -> tuple[list[PreparedTask], list[dict[str, object]]]:
    """Reads SGFs and constructs one bounded-concurrency engine batch."""
    prepared_tasks: list[PreparedTask] = []
    queries: list[dict[str, object]] = []
    for batch_offset, task in enumerate(tasks):
        row = data.read_game_row(references[task.reference_index])
        moves = katago.parse_sgf_moves(row["sgf_content"])
        player_color = row["color"]
        turns = katago.sample_player_turns(
            moves,
            player_color,
            extraction_config.samples_per_game,
        )
        request_id = str(start_index + batch_offset)
        prepared_tasks.append(
            PreparedTask(task, moves, turns, player_color, request_id),
        )
        if not extraction_config.human_features_only:
            queries.append(
                katago.make_analysis_query(
                    request_id,
                    moves,
                    turns,
                    extraction_config.max_visits,
                    katago.AnalysisQueryConfig(
                        extraction_config.include_transition_features,
                    ),
                ),
            )
        if extraction_config.human_sl_profiles:
            for profile_index, profile in enumerate(
                extraction_config.human_sl_profiles,
            ):
                queries.append(
                    katago.make_analysis_query(
                        make_human_request_id(request_id, profile_index),
                        moves,
                        turns,
                        extraction_config.max_visits,
                        katago.AnalysisQueryConfig(
                            extraction_config.include_transition_features,
                            profile,
                        ),
                    ),
                )
    return prepared_tasks, queries


def make_human_request_id(request_id: str, profile_index: int) -> str:
    """Builds an engine request ID safe from arbitrary profile name content."""
    return f"{request_id}:human:{profile_index}"


def select_task_responses(
    responses: collections.abc.Mapping[str, list[dict[str, object]]],
    request_id: str,
    human_sl_profiles: tuple[str, ...],
    human_features_only: bool,
) -> tuple[
    list[dict[str, object]],
    dict[str, list[dict[str, object]]] | None,
]:
    """Selects one normal response stream and optional HumanSL streams."""
    if not human_sl_profiles:
        return _response_for_id(responses, request_id), None
    human_responses = {
        profile: _response_for_id(
            responses,
            make_human_request_id(request_id, profile_index),
        )
        for profile_index, profile in enumerate(human_sl_profiles)
    }
    if human_features_only:
        return human_responses[human_sl_profiles[0]], human_responses
    return _response_for_id(responses, request_id), human_responses


def _response_for_id(
    responses: collections.abc.Mapping[str, list[dict[str, object]]],
    request_id: str,
) -> list[dict[str, object]]:
    """Returns responses for one request with a clear protocol error."""
    response = responses.get(request_id)
    if response is None:
        raise ValueError(f"KataGo returned no response for request {request_id!r}")
    return response


def save_features(
    output_path: pathlib.Path,
    name: str,
    feature_data: tuple[np.ndarray, np.ndarray, np.ndarray],
    extraction_config: ExtractionConfig,
) -> None:
    """Persists one candidate or query feature array in compressed NumPy form."""
    tokens, masks, labels = feature_data
    np.savez_compressed(
        output_path / f"{name}.npz",
        tokens=tokens,
        masks=masks,
        player_indices=labels,
        feature_names=np.asarray(
            katago.decision_feature_names(
                extraction_config.include_transition_features,
                extraction_config.human_sl_profiles,
            ),
        ),
    )


def make_holdout_config(arguments: argparse.Namespace) -> data.RetrievalHoldoutConfig:
    """Creates the same deterministic holdout configuration as evaluation."""
    return data.RetrievalHoldoutConfig(
        arguments.games_per_player,
        arguments.candidate_games,
        arguments.query_games,
        arguments.candidate_players,
        arguments.holdout_seed,
        arguments.player_color,
        arguments.exclude_holdout_seed,
    )


def main() -> None:
    """Extracts candidate and query decision tokens using the configured engine."""
    arguments = parse_arguments()
    if arguments.output.exists():
        raise FileExistsError(f"Feature output directory already exists: {arguments.output!s}")
    human_sl_profiles = tuple(arguments.human_sl_profiles)
    validate_human_sl_arguments(
        arguments.human_model,
        human_sl_profiles,
        arguments.human_features_only,
    )
    holdout_config = make_holdout_config(arguments)
    references, holdout = data.load_retrieval_holdout(
        arguments.index_path,
        holdout_config,
    )
    extraction_config = ExtractionConfig(
        arguments.samples_per_game,
        arguments.max_visits,
        arguments.games_per_engine_batch,
        arguments.include_transition_features,
        human_sl_profiles,
        arguments.human_features_only,
    )
    arguments.output.mkdir(parents=True)
    with katago.KataGoClient(
        arguments.engine,
        arguments.engine_config,
        arguments.model,
        arguments.human_model,
    ) as client:
        if arguments.split == "holdout":
            candidate_tasks, query_tasks = make_tasks(holdout)
            candidate_features = extract_tasks(
                client,
                references,
                candidate_tasks,
                extraction_config,
            )
            query_features = extract_tasks(
                client,
                references,
                query_tasks,
                extraction_config,
            )
            save_features(
                arguments.output,
                "candidates",
                candidate_features,
                extraction_config,
            )
            save_features(
                arguments.output,
                "queries",
                query_features,
                extraction_config,
            )
        else:
            train_tasks = make_train_tasks(
                holdout,
                arguments.train_games_per_player,
                arguments.holdout_seed,
            )
            train_features = extract_tasks(
                client,
                references,
                train_tasks,
                extraction_config,
            )
            save_features(
                arguments.output,
                "train",
                train_features,
                extraction_config,
            )
    metadata = {
        "candidate_players": arguments.candidate_players,
        "candidate_games": arguments.candidate_games,
        "query_games": arguments.query_games,
        "samples_per_game": arguments.samples_per_game,
        "max_visits": arguments.max_visits,
        "holdout_seed": arguments.holdout_seed,
        "exclude_holdout_seed": arguments.exclude_holdout_seed,
        "player_color": arguments.player_color,
        "split": arguments.split,
        "train_games_per_player": arguments.train_games_per_player,
        "include_transition_features": arguments.include_transition_features,
        "model": str(arguments.model),
        "human_model": str(arguments.human_model) if arguments.human_model else None,
        "human_sl_profiles": human_sl_profiles,
        "human_features_only": arguments.human_features_only,
        "features": katago.decision_feature_names(
            arguments.include_transition_features,
            human_sl_profiles,
        ),
    }
    with (arguments.output / "metadata.json").open("w", encoding="utf-8") as output_file:
        json.dump(metadata, output_file, indent=2)


def validate_human_sl_arguments(
    human_model_path: pathlib.Path | None,
    human_sl_profiles: tuple[str, ...],
    human_features_only: bool,
) -> None:
    """Ensures HumanSL feature extraction has one model and profiles together."""
    katago.make_human_policy_feature_names(human_sl_profiles)
    if human_sl_profiles and human_model_path is None:
        raise ValueError("--human-sl-profiles requires --human-model")
    if human_model_path is not None and not human_sl_profiles:
        raise ValueError("--human-model requires --human-sl-profiles")
    if human_features_only and not human_sl_profiles:
        raise ValueError("--human-features-only requires --human-sl-profiles")
    if human_model_path is not None and not human_model_path.is_file():
        raise FileNotFoundError(f"HumanSL model does not exist: {human_model_path!s}")


if __name__ == "__main__":
    main()
