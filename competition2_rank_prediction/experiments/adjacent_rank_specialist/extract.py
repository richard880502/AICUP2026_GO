"""Extracts excluded-half game embeddings for the adjacent-rank pilot.

Run the companion ``run`` module after this module, with the repository's
competition2_rank_prediction directory in PYTHONPATH.
"""

import hashlib
import json
import pathlib
import dataclasses

import numpy as np
import torch

import rank_lib  # pylint: disable=import-error
import transformer_rank  # pylint: disable=import-error
from competition2_rank_prediction.experiments.rank_experts import common
from competition2_rank_prediction.experiments.rank_experts import run as pilot_run


@dataclasses.dataclass(frozen=True)
class ExtractionContext:
    """Stores aligned inputs for one embedding extraction run.

    Attributes:
        root: Research workspace containing features and checkpoints.
        data: Audited question-level metadata.
        keys: Game keys in evaluation order.
        rows: Question index corresponding to each game.
        device: Torch inference device.
    """

    root: pathlib.Path
    data: pilot_run.PilotData
    keys: np.ndarray
    rows: np.ndarray
    device: torch.device


def _encode(
    checkpoint: pathlib.Path,
    features: dict[str, np.ndarray],
    selected: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arguments = state["args"]
    tokens = transformer_rank.decision_tokens(features["tokens"][selected])
    valid = features["masks"][selected].copy()
    empty = ~valid.any(axis=1)
    valid[empty, 0] = True
    tokens = np.where(
        valid[..., None], (tokens - state["mean"]) / state["std"], 0,
    ).astype(np.float32)
    model = transformer_rank.RankTransformer(
        tokens.shape[2], tokens.shape[1], arguments["d_model"],
        arguments["heads"], arguments["ffn"], arguments["layers"],
        arguments["dropout"], arguments.get("head", "clm"),
    ).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    result = _predict_embeddings(model, tokens, valid, device)
    result[empty] = 0
    return result


def _predict_embeddings(
    model: torch.nn.Module,
    tokens: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(tokens), 256):
            chunk = torch.from_numpy(tokens[start:start + 256]).to(device)
            mask = torch.from_numpy(valid[start:start + 256]).to(device)
            outputs.append(model.encode(chunk, mask).cpu().numpy())
    return np.concatenate(outputs)


def _game_rows(root: pathlib.Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = root / "experiments/sgf_structure/results"
    with np.load(source / "base_holdout3.npz") as payload:
        questions = np.asarray(payload["question_id"])
    with np.load(source / "game_members_holdout3.npz") as payload:
        game_keys = np.asarray(payload["game_key"])
        game_questions = np.asarray(payload["question_id"])
    question_index = {str(key): index for index, key in enumerate(questions)}
    rows = np.asarray([question_index[str(key)] for key in game_questions])
    return questions, game_keys, rows


def _extract_side(
    context: ExtractionContext,
    directory: str,
    prefix: str,
) -> tuple[np.ndarray, dict[str, str]]:
    features = rank_lib.load_features(
        context.root / "features" / directory / "holdout3",
    )
    lookup = {str(key): index for index, key in enumerate(features["game_key"])}
    aligned = np.asarray([lookup[str(key)] for key in context.keys])
    pooled = np.zeros((len(context.data.players), 128), dtype=np.float32)
    hashes = {}
    for half in (0, 1):
        game_rows = np.flatnonzero(context.data.halves[context.rows] == half)
        checkpoint = context.root / "results" / f"{prefix}_c{half}_seed0.pt"
        hashes[checkpoint.name] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        embeddings = _encode(checkpoint, features, aligned[game_rows], context.device)
        if embeddings.shape[1] != 128:
            raise ValueError("Unexpected encoder embedding width")
        for question_row in np.unique(context.rows[game_rows]):
            mask = context.rows[game_rows] == question_row
            pooled[question_row] = embeddings[mask].mean(axis=0)
        print(f"{prefix} half={half}: {len(game_rows)} games", flush=True)
    return pooled, hashes


def main() -> None:
    """Checks exclusions and writes aligned question embeddings."""
    arguments = common.parse_arguments(
        __doc__ or "", "adjacent_rank_specialist_20261107", True,
    )
    if arguments.output.exists():
        raise FileExistsError(f"Output already exists: {arguments.output}")
    data = pilot_run.load_data(arguments.research)
    questions, game_keys, game_question_rows = _game_rows(arguments.research)
    pooled = np.zeros((len(questions), 256), dtype=np.float32)
    hashes = {}
    device = torch.device(arguments.device)
    context = ExtractionContext(
        arguments.research, data, game_keys, game_question_rows, device,
    )
    for side, directory, prefix in (
        (0, "feat96", "d96"), (1, "feat96opp", "opp_scratch"),
    ):
        side_values, side_hashes = _extract_side(
            context, directory, prefix,
        )
        pooled[:, side * 128:(side + 1) * 128] = side_values
        hashes.update(side_hashes)
    arguments.output.mkdir(parents=True)
    np.savez_compressed(
        arguments.output / "embeddings.npz", embeddings=pooled,
        players=data.players, labels=data.labels, halves=data.halves,
    )
    (arguments.output / "checkpoints.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
