# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Checks real last-block gradients and GPU cost before any rank experiment.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Uses training positions and synthetic labels, discarding all updated weights.
"""

import csv
import dataclasses
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import equivalence
from competition2_rank_prediction.experiments.rank_experts import common

# The research modules live in the project root, which must be on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import extract_rank_features  # pylint: disable=wrong-import-position,import-error


@dataclasses.dataclass
class GradientOutputs:
    """Captures only trunkfinal without the upstream diagnostic detach.

    Attributes:
        returned: Named tensors retaining their original autograd graph.
    """

    returned: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)

    def report(self, name: str, value: torch.Tensor) -> None:
        """Keeps the requested trunk tensor and ignores other diagnostics."""
        if name == "trunkfinal":
            self.returned[name] = value


def training_inputs(args, modules, payload) -> list[torch.Tensor]:
    """Builds four training games with eight own decisions and neutral metadata."""
    with (args.research / "features/shards/train/games_0000.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        games = list(itertools.islice(csv.DictReader(handle), 4))
    arrays: list[list[np.ndarray]] = [[], [], []]
    points = []
    for game in games:
        moves = extract_rank_features.parse_moves(game["sgf_content"])
        turns = extract_rank_features.katago.sample_player_turns(
            moves, game["color"], 8
        )
        if len(turns) != 8:
            raise ValueError("Probe requires eight decisions per game")
        for destination, values in zip(
            arrays,
            equivalence.make_inputs(modules, payload["config"], moves, turns),
        ):
            destination.append(values)
        for turn in turns:
            move = moves[turn]
            points.append(361 if move.x is None else move.y * 19 + move.x)
    tensors = [
        torch.as_tensor(np.concatenate(values), device=args.device)
        for values in arrays
    ]
    tensors.append(torch.as_tensor(points, device=args.device))
    return tensors


def representation(model, tensors: list[torch.Tensor]) -> torch.Tensor:
    """Returns per-game mean global/played-point trunk features with gradients."""
    captured = GradientOutputs()
    model(*tensors[:3], extra_outputs=captured)
    trunk = captured.returned["trunkfinal"]
    points = tensors[3]
    local = trunk.flatten(2).transpose(1, 2)[
        torch.arange(len(points), device=points.device), points.clamp(max=360)
    ]
    local = torch.where((points < 361)[:, None], local, 0)
    features = torch.cat((trunk.mean((2, 3)), local), dim=1)
    return features.reshape(4, 8, 768).mean(dim=1)


def measure(model, tensors: list[torch.Tensor], device: str) -> dict:
    """Performs synthetic updates and verifies frozen-prefix parameter isolation."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if name.startswith(("blocks.16.", "blocks.17.", "norm_trunkfinal.")):
            parameter.requires_grad_(True)
    head = torch.nn.Linear(768, 10).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [*trainable, *head.parameters()], lr=1e-5, weight_decay=0.01
    )
    if not trainable:
        raise ValueError("No last-block trainable parameters selected")
    before = {
        "last": trainable[0].detach().clone(),
        "prefix": model.conv_spatial.weight.detach().clone(),
    }
    torch.cuda.reset_peak_memory_stats(device=device)
    started = time.perf_counter()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = head(representation(model, tensors))
        loss = torch.nn.functional.cross_entropy(
            output, torch.arange(4, device=device)
        )
        loss.backward()
        if (
            trainable[0].grad is None
            or not torch.isfinite(trainable[0].grad).all()
        ):
            raise ValueError("Trunk gradient missing or nonfinite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device=device)
    changed = not torch.equal(before["last"], trainable[0])
    frozen = torch.equal(before["prefix"], model.conv_spatial.weight)
    if not changed or not frozen:
        raise ValueError(
            "Incorrect last-block update or frozen-prefix mutation"
        )
    return {
        "synthetic_losses_not_rank_scores": losses,
        "seconds_for_three_updates": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device=device),
        "trainable_backbone_parameters": sum(p.numel() for p in trainable),
        "last_block_parameter_changed": changed,
        "frozen_prefix_unchanged": frozen,
        "positions_per_update": 32,
        "games_per_update": 4,
        "weights_saved": False,
    }


def main() -> None:
    """Writes isolated feasibility results without saving adapted model weights."""
    args = common.parse_arguments(
        __doc__ or "", "human_finetune_probe_20261109", True
    )
    if args.device != "cuda:1":
        raise ValueError("Only GPU1 is in scope for the adaptation probe")
    if args.output.exists():
        raise FileExistsError(f"Refusing existing probe output: {args.output}")
    modules, payload = bridge.load_assets(args.research)
    model = bridge.build_model(modules, payload, args.device, False)
    # Keep normalization statistics frozen; parameter gradients still propagate.
    model.eval()
    result = measure(
        model, training_inputs(args, modules, payload), args.device
    )
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
