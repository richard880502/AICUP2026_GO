# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Checks HumanSL PyTorch policies against the installed official engine.

Example: python -m competition2_rank_prediction.experiments.human_trunk.equivalence
Only a fixed validation game's positions are used; no rank-score evaluation.
"""

import csv
import json
import pathlib
import sys
import tempfile
import types
from typing import Any

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import smoke
from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


def make_inputs(
    modules: dict[str, types.ModuleType],
    config: dict[str, Any],
    moves: list,
    turns: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replays SGF moves with the upstream board/history feature implementation."""
    state = modules["gamestate"].GameState(
        19, modules["gamestate"].GameState.RULES_JAPANESE
    )
    feature_builder = modules["features"].Features(config, 19)
    spatial, global_features, meta = [], [], []
    for index, move in enumerate(moves):
        if index in turns:
            binary, global_row = state.get_input_features(feature_builder)
            spatial.append(binary[0])
            global_features.append(global_row[0])
            meta.append(
                smoke.neutral_metadata(modules).get_metadata_row(
                    state.board.pla, 361
                )
            )
        player = state.board.BLACK if move.color == "B" else state.board.WHITE
        location = (
            state.board.PASS_LOC
            if move.x is None
            else state.board.loc(move.x, move.y)
        )
        state.play(player, location)
    if len(spatial) != len(turns):
        raise ValueError("Did not construct every requested pre-move position")
    return np.stack(spatial), np.stack(global_features), np.stack(meta)


def engine_policies(args, moves: list, turns: list[int]) -> np.ndarray:
    """Queries unmodified HumanSL1d with deterministic engine symmetry."""
    if args.device != "cuda:1":
        raise ValueError("Equivalence engine is restricted to available GPU1")
    models = args.research / "c1/models/katago"
    query = extract_rank_features.katago.make_analysis_query(
        "equivalence",
        moves,
        turns,
        1,
        extract_rank_features.katago.AnalysisQueryConfig(False, "rank_1d"),
    )
    with tempfile.TemporaryDirectory(
        prefix="aicup_humansl_equivalence_"
    ) as temporary:
        config = pathlib.Path(temporary) / "analysis.cfg"
        template = (
            args.research / "configs/analysis_fast.cfg.tmpl"
        ).read_text()
        config.write_text(
            template.replace("@DEVICE@", "1") + "\nnnRandomize = false\n"
        )
        with extract_rank_features.katago.KataGoClient(
            args.research / "c1/scripts/run_katago.sh",
            config,
            models / "kata1-b18c384nbt-s9996604416-d4316597426.bin.gz",
            models / "b18c384nbt-humanv0.bin.gz",
        ) as client:
            responses = client.analyze([query])["equivalence"]
    lookup = {int(row["turnNumber"]): row["humanPolicy"] for row in responses}
    return np.asarray([lookup[turn] for turn in turns], dtype=np.float64)


def compare(policy: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    """Applies engine legal-move support before comparing normalized raw policy."""
    if policy.ndim != 2 or policy.shape != reference.shape or len(policy) == 0:
        raise ValueError("Expected matching nonempty two-dimensional policies")
    if not np.isfinite(policy).all() or not np.isfinite(reference).all():
        raise ValueError("Policy comparison requires finite values")
    predicted = np.where(reference >= 0, policy, 0)
    if (predicted.sum(axis=1) <= 0).any():
        raise ValueError("Policy has no positive mass on legal moves")
    predicted /= predicted.sum(axis=1, keepdims=True)
    expected = np.maximum(reference, 0)
    if (expected.sum(axis=1) <= 0).any():
        raise ValueError("Engine reference has no legal policy mass")
    expected /= expected.sum(axis=1, keepdims=True)
    distance = np.abs(predicted - expected)
    variation = distance.sum(axis=1) / 2
    return {
        "mean_total_variation": float(variation.mean()),
        "max_total_variation": float(variation.max()),
        "max_absolute_error": float(distance.max()),
        "per_position_total_variation": variation.tolist(),
        "passes_tolerance": float(variation.mean()) <= 0.005
        and float(variation.max()) <= 0.02,
    }


def main() -> None:
    """Compares raw/SWA solely to identify export compatibility, never rank utility."""
    args = common.parse_arguments(__doc__ or "", "human_trunk_20261109", True)
    if (args.output / "equivalence.json").exists():
        raise FileExistsError("Refusing to overwrite equivalence result")
    torch.set_num_threads(4)
    with (args.research / "features/shards/val/games_0000.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        game = next(csv.DictReader(handle))
    moves = extract_rank_features.parse_moves(game["sgf_content"])
    turns = extract_rank_features.katago.sample_player_turns(
        moves, game["color"], 8
    )
    modules = bridge.load_vendor(
        args.research / "vendor/KataGo-human-v1.15.0/python"
    )
    payload = bridge.checkpoint(
        args.research / "models/human_trunk/b18c384nbt-humanv0.ckpt"
    )
    inputs = make_inputs(modules, payload["config"], moves, turns)
    reference = engine_policies(args, moves, turns)
    report = {}
    for use_swa in (False, True):
        model = bridge.build_model(modules, payload, args.device, use_swa)
        with torch.inference_mode():
            output = model(
                *[
                    torch.as_tensor(values, device=args.device)
                    for values in inputs
                ]
            )
            policy = output[0][0][:, 0].softmax(dim=-1).cpu().numpy()
        report["swa" if use_swa else "raw"] = compare(policy, reference)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "equivalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
