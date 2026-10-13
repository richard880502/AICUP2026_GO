"""KataGo analysis-engine client and decision-token construction.

The token describes a player's actual move relative to KataGo's raw policy at
the position immediately before that move. It is deliberately compact so that
features can be cached for every game and reused by multiple encoders.
"""

import collections.abc
import dataclasses
import json
import math
import pathlib
import queue
import subprocess
import threading
import typing

import numpy as np

from go_retrieval import sgf


KOMI = 6.5
BASE_DECISION_FEATURE_NAMES = (
    "policy_surprisal",
    "log_policy_rank",
    "policy_entropy",
    "player_winrate",
    "player_score_lead_scaled",
    "x_normalized",
    "y_normalized",
    "game_phase",
)

TRANSITION_DECISION_FEATURE_NAMES = (
    "winrate_drop",
    "score_lead_drop_scaled",
    "ownership_change",
    "spatial_ownership_certainty",
    "spatial_ownership_move_pre",
    "spatial_ownership_global_gain",
    "spatial_ownership_move_gain",
    "spatial_ownership_local_change_share_r1",
    "spatial_ownership_local_change_share_r2",
    "spatial_ownership_change_mean_distance",
    "spatial_ownership_change_concentration",
    "spatial_ownership_change_entropy",
    "spatial_ownership_net_change_ratio",
    "spatial_ownership_change_direction_x",
    "spatial_ownership_change_direction_y",
    "spatial_ownership_change_top_left",
    "spatial_ownership_change_top_center",
    "spatial_ownership_change_top_right",
    "spatial_ownership_change_middle_left",
    "spatial_ownership_change_middle_center",
    "spatial_ownership_change_middle_right",
    "spatial_ownership_change_bottom_left",
    "spatial_ownership_change_bottom_center",
    "spatial_ownership_change_bottom_right",
)


def decision_feature_names(
    include_transition_features: bool,
    human_sl_profiles: collections.abc.Sequence[str] = (),
) -> tuple[str, ...]:
    """Returns the ordered feature schema for a decision-token cache."""
    human_feature_names = make_human_policy_feature_names(human_sl_profiles)
    if include_transition_features:
        return (
            BASE_DECISION_FEATURE_NAMES
            + human_feature_names
            + TRANSITION_DECISION_FEATURE_NAMES
        )
    return BASE_DECISION_FEATURE_NAMES + human_feature_names


def make_human_policy_feature_names(
    human_sl_profiles: collections.abc.Sequence[str],
) -> tuple[str, ...]:
    """Returns per-profile surprisal, rank, and entropy feature names."""
    _validate_human_sl_profiles(human_sl_profiles)
    names: list[str] = []
    for profile in human_sl_profiles:
        names.extend(
            (
                f"human_{profile}_surprisal",
                f"human_{profile}_log_policy_rank",
                f"human_{profile}_policy_entropy",
            ),
        )
    return tuple(names)


@dataclasses.dataclass(frozen=True)
class SgfMove:
    """A parsed SGF move with zero-based board coordinates.

    Attributes:
        color: ``B`` or ``W``.
        x: Column from the left, or ``None`` for pass.
        y: Row from the top, or ``None`` for pass.
    """

    color: str
    x: int | None
    y: int | None


@dataclasses.dataclass(frozen=True)
class AnalysisQueryConfig:
    """Optional model settings applied to one analysis-engine query."""

    include_transition_features: bool = False
    human_sl_profile: str | None = None


@dataclasses.dataclass(frozen=True)
class DecisionResponses:
    """Normal-model responses plus optional HumanSL response streams."""

    normal: collections.abc.Sequence[dict[str, object]]
    human_by_profile: collections.abc.Mapping[
        str,
        collections.abc.Sequence[dict[str, object]],
    ] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DecisionTokenContext:
    """Indexed engine responses and settings shared by sampled turns."""

    moves: collections.abc.Sequence[SgfMove]
    player_color: str
    normal_by_turn: dict[int, dict[str, object]]
    human_by_profile: dict[str, dict[int, dict[str, object]]]
    include_transition_features: bool


class KataGoClient:
    """Streams batches of analysis queries to one persistent KataGo process."""

    def __init__(
        self,
        executable_path: pathlib.Path,
        config_path: pathlib.Path,
        model_path: pathlib.Path,
        human_model_path: pathlib.Path | None = None,
    ):
        self.executable_path = executable_path
        self.config_path = config_path
        self.model_path = model_path
        self.human_model_path = human_model_path
        self.process: subprocess.Popen[str] | None = None
        self.response_queue: queue.Queue[str | None] | None = None
        self.output_thread: threading.Thread | None = None

    def __enter__(self) -> "KataGoClient":
        """Starts KataGo's JSON-lines analysis engine."""
        command = [
            str(self.executable_path),
            "analysis",
            "-config",
            str(self.config_path),
            "-model",
            str(self.model_path),
        ]
        if self.human_model_path is not None:
            command.extend(("-human-model", str(self.human_model_path)))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self.process.stdout is None:
            raise RuntimeError("KataGo stdout pipe was not created")
        self.response_queue = queue.Queue()
        self.output_thread = threading.Thread(
            target=read_katago_output,
            args=(self.process.stdout, self.response_queue),
            daemon=True,
        )
        self.output_thread.start()
        return self

    def __exit__(self, exception_type: object, exception: object, traceback: object) -> None:
        """Stops KataGo even when an analysis response is malformed."""
        del exception_type, exception, traceback
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=30)
            self.process = None
        if self.output_thread is not None:
            self.output_thread.join(timeout=5)
            self.output_thread = None
        self.response_queue = None

    def analyze(
        self,
        queries: collections.abc.Sequence[dict[str, object]],
    ) -> dict[str, list[dict[str, object]]]:
        """Submits queries and returns terminal responses keyed by request ID."""
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("KataGoClient must be used as a context manager")
        if self.response_queue is None:
            raise RuntimeError("KataGo response reader is unavailable")
        for query in queries:
            self.process.stdin.write(json.dumps(query) + "\n")
        self.process.stdin.flush()
        responses: dict[str, list[dict[str, object]]] = {}
        remaining = 0
        for query in queries:
            turns = query.get("analyzeTurns", [])
            if not isinstance(turns, list):
                raise ValueError("KataGo query analyzeTurns must be a list")
            remaining += len(turns)
        while remaining:
            output_line = self.response_queue.get()
            if output_line is None:
                raise RuntimeError("KataGo exited before returning every requested result")
            response = json.loads(output_line)
            if "error" in response:
                raise RuntimeError(f"KataGo analysis error: {response['error']}")
            if response.get("isDuringSearch"):
                continue
            request_id = str(response["id"])
            responses.setdefault(request_id, []).append(response)
            remaining -= 1
        return responses


def read_katago_output(
    output_stream: typing.TextIO,
    response_queue: queue.Queue[str | None],
) -> None:
    """Streams KataGo output so a full stdout pipe cannot block stdin writes."""
    try:
        for output_line in output_stream:
            response_queue.put(output_line)
    finally:
        response_queue.put(None)


def parse_sgf_moves(sgf_content: str) -> list[SgfMove]:
    """Parses all non-setup 19x19 SGF moves, retaining pass moves."""
    moves: list[SgfMove] = []
    for color, coordinate in sgf.MOVE_PATTERN.findall(sgf_content):
        if coordinate:
            moves.append(
                SgfMove(
                    color=color,
                    x=ord(coordinate[0]) - ord("a"),
                    y=ord(coordinate[1]) - ord("a"),
                )
            )
        else:
            moves.append(SgfMove(color=color, x=None, y=None))
    return moves


def sgf_move_to_gtp(move: SgfMove) -> str:
    """Converts an SGF coordinate to a 19x19 GTP coordinate."""
    if move.x is None or move.y is None:
        return "pass"
    column_offset = move.x + (1 if move.x >= 8 else 0)
    column = chr(ord("A") + column_offset)
    return f"{column}{sgf.BOARD_SIZE - move.y}"


def sample_player_turns(
    moves: collections.abc.Sequence[SgfMove],
    player_color: str,
    samples_per_game: int,
) -> list[int]:
    """Selects evenly spaced non-pass decisions made by one player."""
    eligible_turns = [
        turn
        for turn, move in enumerate(moves)
        if move.color == player_color and move.x is not None
    ]
    if not eligible_turns:
        return []
    sample_count = min(samples_per_game, len(eligible_turns))
    selected_offsets = np.linspace(0, len(eligible_turns) - 1, sample_count, dtype=int)
    return [eligible_turns[offset] for offset in selected_offsets]


def make_analysis_query(
    request_id: str,
    moves: collections.abc.Sequence[SgfMove],
    turns: collections.abc.Sequence[int],
    max_visits: int,
    analysis_config: AnalysisQueryConfig | None = None,
) -> dict[str, object]:
    """Builds a raw-policy KataGo request for the chosen pre-move positions."""
    if analysis_config is None:
        analysis_config = AnalysisQueryConfig()
    analysis_turns = list(turns)
    if analysis_config.include_transition_features:
        analysis_turns.extend(min(turn + 1, len(moves)) for turn in turns)
    query: dict[str, object] = {
        "id": request_id,
        "moves": [[move.color, sgf_move_to_gtp(move)] for move in moves],
        "rules": "japanese",
        "komi": KOMI,
        "boardXSize": sgf.BOARD_SIZE,
        "boardYSize": sgf.BOARD_SIZE,
        "analyzeTurns": list(dict.fromkeys(analysis_turns)),
        "maxVisits": max_visits,
        "analysisPVLen": 1,
        "includePolicy": True,
        "includeOwnership": analysis_config.include_transition_features,
    }
    if analysis_config.human_sl_profile is not None:
        _validate_human_sl_profiles((analysis_config.human_sl_profile,))
        query["overrideSettings"] = {
            "humanSLProfile": analysis_config.human_sl_profile,
            "ignorePreRootHistory": False,
        }
    return query


def build_decision_tokens(
    moves: collections.abc.Sequence[SgfMove],
    player_color: str,
    turns: collections.abc.Sequence[int],
    responses: DecisionResponses,
    include_transition_features: bool = False,
) -> np.ndarray:
    """Builds fixed-order decision tokens from KataGo policy responses."""
    context = DecisionTokenContext(
        moves,
        player_color,
        make_response_by_turn(responses.normal),
        make_human_response_by_profile(responses.human_by_profile),
        include_transition_features,
    )
    tokens: list[list[float]] = []
    for turn in turns:
        token = make_token_for_turn(turn, context)
        if token is not None:
            tokens.append(token)
    return np.asarray(tokens, dtype=np.float32)


def make_response_by_turn(
    responses: collections.abc.Sequence[dict[str, object]],
) -> dict[int, dict[str, object]]:
    """Indexes final KataGo responses by the analyzed turn number."""
    response_by_turn: dict[int, dict[str, object]] = {}
    for response in responses:
        turn_number = response.get("turnNumber")
        if not isinstance(turn_number, int):
            raise ValueError("KataGo response has no integer turnNumber")
        response_by_turn[turn_number] = response
    return response_by_turn


def make_human_response_by_profile(
    responses_by_profile: collections.abc.Mapping[
        str,
        collections.abc.Sequence[dict[str, object]],
    ],
) -> dict[str, dict[int, dict[str, object]]]:
    """Indexes each requested HumanSL policy stream by move turn."""
    return {
        profile: make_response_by_turn(responses)
        for profile, responses in responses_by_profile.items()
    }


def make_token_for_turn(
    turn: int,
    context: DecisionTokenContext,
) -> list[float] | None:
    """Builds all normal and HumanSL features for one sampled move."""
    turn_response = context.normal_by_turn.get(turn)
    if turn_response is None:
        raise ValueError(f"KataGo returned no result for turn {turn}")
    move = context.moves[turn]
    if move.x is None or move.y is None:
        return None
    phase = turn / max(len(context.moves) - 1, 1)
    token = make_decision_token(
        move,
        context.player_color,
        phase,
        turn_response,
    )
    for human_response_by_turn in context.human_by_profile.values():
        human_response = human_response_by_turn.get(turn)
        if human_response is None:
            raise ValueError(
                f"KataGo returned no human-policy result for turn {turn}",
            )
        token.extend(make_human_policy_features(move, human_response))
    if context.include_transition_features:
        post_move_response = context.normal_by_turn.get(turn + 1)
        if post_move_response is None:
            raise ValueError(f"KataGo returned no post-move result for turn {turn}")
        token.extend(
            make_transition_features(
                move,
                context.player_color,
                turn_response,
                post_move_response,
            ),
        )
    return token


def make_decision_token(
    move: SgfMove,
    player_color: str,
    phase: float,
    response: dict[str, object],
) -> list[float]:
    """Converts one engine response and human move into scalar features."""
    if move.x is None or move.y is None:
        raise ValueError("Decision token requires a non-pass move")
    policy_values = response.get("policy")
    root_info = response.get("rootInfo")
    if not isinstance(policy_values, list) or not isinstance(root_info, dict):
        raise ValueError("KataGo response is missing policy or rootInfo")
    policy_features = make_policy_features(move, policy_values, "policy")
    winrate = _numeric_root_value(root_info, "rawWinrate")
    score_lead = _numeric_root_value(root_info, "rawLead")
    if player_color == "W":
        winrate = 1.0 - winrate
        score_lead = -score_lead
    return [
        *policy_features,
        winrate,
        float(np.clip(score_lead / 20.0, -5.0, 5.0)),
        move.x / (sgf.BOARD_SIZE - 1),
        move.y / (sgf.BOARD_SIZE - 1),
        phase,
    ]


def make_human_policy_features(
    move: SgfMove,
    response: dict[str, object],
) -> list[float]:
    """Measures how natural a played move is to one HumanSL profile."""
    human_policy = response.get("humanPolicy")
    if not isinstance(human_policy, list):
        raise ValueError("KataGo response is missing humanPolicy")
    return make_policy_features(move, human_policy, "humanPolicy")


def make_policy_features(
    move: SgfMove,
    policy_values: list[object],
    field_name: str,
) -> list[float]:
    """Computes surprisal, rank, and entropy for one legal played move."""
    if move.x is None or move.y is None:
        raise ValueError("Policy features require a non-pass move")
    policy = np.asarray(policy_values, dtype=np.float64)
    policy_index = move.y * sgf.BOARD_SIZE + move.x
    if policy_index >= len(policy):
        raise ValueError(f"KataGo {field_name} has an invalid board shape")
    probability = float(policy[policy_index])
    if probability < 0:
        raise ValueError(f"Played move is illegal under KataGo {field_name}")
    probability = max(probability, 1e-12)
    legal_policy = policy[policy >= 0]
    if not legal_policy.size:
        raise ValueError(f"KataGo {field_name} has no legal moves")
    policy_rank = int(np.sum(legal_policy > probability)) + 1
    entropy = -float(
        np.sum(legal_policy * np.log(np.maximum(legal_policy, 1e-12))),
    )
    return [-math.log(probability), math.log1p(policy_rank), entropy]


def make_transition_features(
    move: SgfMove,
    player_color: str,
    pre_move_response: dict[str, object],
    post_move_response: dict[str, object],
) -> list[float]:
    """Measures value, score, and ownership changes caused by the human move."""
    pre_root = _root_info(pre_move_response)
    post_root = _root_info(post_move_response)
    pre_winrate, pre_score = _player_values(pre_root, player_color)
    post_winrate, post_score = _player_values(post_root, player_color)
    pre_ownership = _ownership_values(pre_move_response)
    post_ownership = _ownership_values(post_move_response)
    if pre_ownership.shape != post_ownership.shape:
        raise ValueError("KataGo ownership sizes differ before and after a move")
    transition_features = [
        pre_winrate - post_winrate,
        float(np.clip((pre_score - post_score) / 20.0, -5.0, 5.0)),
        float(np.mean(np.abs(pre_ownership - post_ownership))),
    ]
    transition_features.extend(make_spatial_ownership_features(
        move,
        player_color,
        pre_ownership,
        post_ownership,
    ))
    return transition_features


def make_spatial_ownership_features(
    move: SgfMove,
    player_color: str,
    pre_ownership: np.ndarray,
    post_ownership: np.ndarray,
) -> list[float]:
    """Describes where a move changes ownership, from the player's view."""
    if move.x is None or move.y is None:
        raise ValueError("Spatial ownership features require a board move")
    if len(pre_ownership) != sgf.BOARD_SIZE ** 2:
        raise ValueError("KataGo ownership has an unexpected board size")
    player_sign = 1.0 if player_color == "B" else -1.0
    player_pre = player_sign * pre_ownership
    ownership_gain = player_sign * post_ownership - player_pre
    absolute_change = np.abs(ownership_gain).reshape(
        sgf.BOARD_SIZE,
        sgf.BOARD_SIZE,
    )
    move_index = move.y * sgf.BOARD_SIZE + move.x
    return [
        float(np.mean(np.abs(player_pre))),
        float(player_pre[move_index]),
        float(np.mean(ownership_gain)),
        float(ownership_gain[move_index]),
        *make_ownership_locality_features(
            ownership_gain.reshape(sgf.BOARD_SIZE, sgf.BOARD_SIZE),
            move,
        ),
        *make_regional_ownership_changes(absolute_change),
    ]


def make_ownership_locality_features(
    ownership_gain: np.ndarray,
    move: SgfMove,
) -> list[float]:
    """Summarizes how concentrated and local an ownership transition is."""
    if move.x is None or move.y is None:
        raise ValueError("Ownership locality features require a board move")
    absolute_change = np.abs(ownership_gain)
    total_change = float(absolute_change.sum())
    if total_change <= 1e-12:
        return [0.0] * 8
    row_coordinates, column_coordinates = np.indices(absolute_change.shape)
    delta_x = column_coordinates - move.x
    delta_y = row_coordinates - move.y
    distances = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    probabilities = absolute_change / total_change
    normalized_distance = math.sqrt(2.0) * (sgf.BOARD_SIZE - 1)
    normalized_entropy = math.log(absolute_change.size)
    return [
        float(probabilities[distances <= 1.0].sum()),
        float(probabilities[distances <= 2.0].sum()),
        float(np.sum(probabilities * distances) / normalized_distance),
        float(np.sum(probabilities * probabilities) * absolute_change.size),
        float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
              / normalized_entropy),
        float(np.mean(ownership_gain) / np.mean(absolute_change)),
        float(np.sum(probabilities * delta_x) / (sgf.BOARD_SIZE - 1)),
        float(np.sum(probabilities * delta_y) / (sgf.BOARD_SIZE - 1)),
    ]


def make_regional_ownership_changes(absolute_change: np.ndarray) -> list[float]:
    """Summarizes absolute ownership change over a fixed 3-by-3 board grid."""
    region_changes = []
    boundaries = (0, 7, 13, sgf.BOARD_SIZE)
    for row_start, row_end in zip(
        boundaries[:-1],
        boundaries[1:],
        strict=True,
    ):
        for column_start, column_end in zip(
            boundaries[:-1],
            boundaries[1:],
            strict=True,
        ):
            region_changes.append(float(np.mean(
                absolute_change[row_start:row_end, column_start:column_end],
            )))
    return region_changes


def _root_info(response: dict[str, object]) -> dict[str, object]:
    """Extracts one response root-info object with a clear error message."""
    root_info = response.get("rootInfo")
    if not isinstance(root_info, dict):
        raise ValueError("KataGo response is missing rootInfo")
    return root_info


def _player_values(root_info: dict[str, object], player_color: str) -> tuple[float, float]:
    """Converts KataGo black-perspective root values into player perspective."""
    winrate = _numeric_root_value(root_info, "rawWinrate")
    score_lead = _numeric_root_value(root_info, "rawLead")
    if player_color == "W":
        return 1.0 - winrate, -score_lead
    return winrate, score_lead


def _ownership_values(response: dict[str, object]) -> np.ndarray:
    """Loads the required ownership vector as a numeric NumPy array."""
    ownership = response.get("ownership")
    if not isinstance(ownership, list):
        raise ValueError("KataGo response is missing ownership")
    return np.asarray(ownership, dtype=np.float64)


def _numeric_root_value(
    root_info: collections.abc.Mapping[str, object],
    name: str,
) -> float:
    """Returns a required numeric root field with a useful error message."""
    value = root_info.get(name)
    if not isinstance(value, int | float):
        raise ValueError(f"KataGo rootInfo has no numeric {name}")
    return float(value)


def _validate_human_sl_profiles(
    human_sl_profiles: collections.abc.Sequence[str],
) -> None:
    """Rejects blank or duplicate HumanSL profile names before extraction."""
    if any(not profile for profile in human_sl_profiles):
        raise ValueError("HumanSL profile names must be non-empty")
    if len(set(human_sl_profiles)) != len(human_sl_profiles):
        raise ValueError("HumanSL profiles must be unique")
