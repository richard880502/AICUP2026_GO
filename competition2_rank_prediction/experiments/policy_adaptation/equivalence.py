# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Compares all candidate-profile policies to the official engine on training games.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Legal-mask equality and established trunk-policy tolerances gate advancement.
"""

import csv
import itertools
import json
import pathlib
import sys
import tempfile

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_finetune import (
    probe as input_probe,
)
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import (
    equivalence as trunk_equivalence,
)
from competition2_rank_prediction.experiments.policy_adaptation import probe
from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def cases(args) -> list:
    """Uses the same four training games/turns as the gradient smoke probe."""
    with (args.research / "features/shards/train/games_0000.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        games = list(itertools.islice(csv.DictReader(handle), 4))
    result = []
    for game in games:
        moves = extract_rank_features.parse_moves(game["sgf_content"])
        turns = extract_rank_features.katago.sample_player_turns(
            moves, game["color"], 8
        )
        result.append((moves, turns))
    return result


def legal_masks(modules, games: list) -> np.ndarray:
    """Replays Japanese simple-ko legality, excluding all suicide moves."""
    masks = []
    for moves, turns in games:
        board = modules["board"].Board(size=19)
        for index, move in enumerate(moves):
            player = board.BLACK if move.color == "B" else board.WHITE
            if index in turns:
                mask = np.ones(362, dtype=bool)
                for point in range(361):
                    location = board.loc(point % 19, point // 19)
                    mask[point] = board.would_be_legal(
                        player, location
                    ) and not board.would_be_suicide(player, location)
                masks.append(mask)
            board.play(
                player,
                board.PASS_LOC if move.x is None else board.loc(move.x, move.y),
            )
    return np.stack(masks)


def reference(args, games: list) -> np.ndarray:
    """Queries every fixed profile for every selected training game."""
    models = args.research / "c1/models/katago"
    queries = []
    for profile in extract_rank_features.PROFILES:
        for index, (moves, turns) in enumerate(games):
            queries.append(
                extract_rank_features.katago.make_analysis_query(
                    f"{profile}_{index}",
                    moves,
                    turns,
                    1,
                    extract_rank_features.katago.AnalysisQueryConfig(
                        False, profile
                    ),
                )
            )
    with tempfile.TemporaryDirectory(
        prefix="aicup_policy_equivalence_"
    ) as temporary:
        config = pathlib.Path(temporary) / "analysis.cfg"
        config.write_text(
            (args.research / "configs/analysis_fast.cfg.tmpl")
            .read_text()
            .replace("@DEVICE@", "1")
            + "\nnnRandomize = false\n"
        )
        with extract_rank_features.katago.KataGoClient(
            args.research / "c1/scripts/run_katago.sh",
            config,
            models / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
            models / "b18c384nbt-humanv0.bin.gz",
        ) as client:
            responses = client.analyze(queries)
    policies: list = []
    for query in queries:
        lookup = {
            int(row["turnNumber"]): row["humanPolicy"]
            for row in responses[query["id"]]
        }
        policies.extend(lookup[turn] for turn in query["analyzeTurns"])
    return np.asarray(policies)


def main() -> None:
    """Writes complete policies for independent replay; no rank scores or training."""
    args = common.parse_arguments(
        __doc__ or "", "policy_equivalence_20261109", True
    )
    if args.device != "cuda:1" or args.output.exists():
        raise ValueError("Requires GPU1 and a new isolated output directory")
    modules, payload = bridge.load_assets(args.research)
    model = bridge.build_model(modules, payload, args.device, False)
    tensors = input_probe.training_inputs(args, modules, payload)
    with torch.inference_mode():
        output = model(
            tensors[0].repeat(10, 1, 1, 1),
            tensors[1].repeat(10, 1),
            input_meta=probe.candidate_metadata(modules, 32, args.device),
        )
        policy = output[0][0][:, 0].softmax(1).cpu().numpy()
    engine = reference(args, cases(args))
    masks = np.tile(legal_masks(modules, cases(args)), (10, 1))
    result = trunk_equivalence.compare(policy, engine)
    result["legal_mask_mismatches"] = int(np.sum(masks != (engine >= 0)))
    result["passes_all_checks"] = bool(
        result["passes_tolerance"] and result["legal_mask_mismatches"] == 0
    )
    args.output.mkdir(parents=True)
    np.savez_compressed(
        args.output / "policies.npz", raw=policy, engine=engine, legal=masks
    )
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                name: value
                for name, value in result.items()
                if name != "per_position_total_variation"
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
