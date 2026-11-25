"""Checks strict weight loading and genuine trunk output on an empty board.

Example: python -m competition2_rank_prediction.experiments.human_trunk.smoke
This is a shape/compatibility test, not a rank-prediction evaluation.
"""

import datetime
import json
import types
from typing import Any

import torch

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.rank_experts import common


def neutral_metadata(modules: dict[str, types.ModuleType]) -> Any:
    """Reproduces the engine rank_1d profile, identically for every sample."""
    return modules["sgfmetadata"].SGFMetadata(
        inverseBRank=9,
        inverseWRank=9,
        bIsHuman=True,
        wIsHuman=True,
        gameRatednessIsUnknown=True,
        tcIsUnknown=False,
        tcIsByoYomi=True,
        mainTimeSeconds=1200,
        periodTimeSeconds=30,
        byoYomiPeriods=5,
        gameDate=datetime.date(2020, 3, 1),
        source=2,
    )


def empty_forward(
    modules: dict[str, types.ModuleType], model: torch.nn.Module, device: str
) -> dict[str, object]:
    """Evaluates upstream board/features/metadata and captures trunkfinal."""
    game = modules["gamestate"].GameState(
        19, modules["gamestate"].GameState.RULES_JAPANESE
    )
    features = modules["features"].Features(model.config, 19)
    spatial, global_features = game.get_input_features(features)
    metadata = neutral_metadata(modules).get_metadata_row(game.board.pla, 361)
    extra = modules["model_pytorch"].ExtraOutputs(["trunkfinal"])
    with torch.inference_mode():
        output = model(
            torch.as_tensor(spatial, device=device),
            torch.as_tensor(global_features, device=device),
            input_meta=torch.as_tensor(metadata[None], device=device),
            extra_outputs=extra,
        )
    trunk = extra.returned["trunkfinal"]
    policy = output[0][0][:, 0].softmax(dim=-1)
    if trunk.shape != (1, 384, 19, 19) or not torch.isfinite(trunk).all():
        raise ValueError("Invalid upstream trunk output")
    return {
        "trunk_shape": list(trunk.shape),
        "trunk_norm": float(trunk.norm().item()),
        "policy_shape": list(policy.shape),
        "policy_sum": float(policy.sum().item()),
        "empty_board_policy": policy[0].cpu().tolist(),
    }


def main() -> None:
    """Writes raw/SWA shape diagnostics without touching production models."""
    args = common.parse_arguments(__doc__ or "", "human_trunk_20261109", True)
    destination = args.output / "smoke.json"
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite smoke test: {destination}"
        )
    modules, payload = bridge.load_assets(args.research)
    results = {}
    for use_swa in (False, True):
        model = bridge.build_model(modules, payload, args.device, use_swa)
        results["swa" if use_swa else "raw"] = empty_forward(
            modules, model, args.device
        )
    args.output.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {name: value["trunk_shape"] for name, value in results.items()}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
