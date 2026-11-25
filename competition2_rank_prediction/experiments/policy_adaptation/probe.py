# Archived experiment; intentionally self-contained (helpers are copied, not shared).
# pylint: disable=duplicate-code
"""Checks differentiable played-move likelihood across all ten candidate ranks.

Run with PYTHONPATH=competition2_rank_prediction python -m plus this path.
Candidate ranks are enumerated for every game, never taken from true labels.
"""

import dataclasses
import json
import time

import numpy as np
import torch

from competition2_rank_prediction.experiments.human_finetune import probe
from competition2_rank_prediction.experiments.human_trunk import bridge
from competition2_rank_prediction.experiments.human_trunk import smoke
from competition2_rank_prediction.experiments.rank_experts import common

INVERSE_RANKS = (20, 17, 14, 11, 9, 8, 7, 6, 5, 4)


def candidate_metadata(modules, positions: int, device: str) -> torch.Tensor:
    """Enumerates fixed11k/8k/5k/2k/1d..6d profiles for every position."""
    rows = []
    for rank in INVERSE_RANKS:
        metadata = dataclasses.replace(
            smoke.neutral_metadata(modules),
            inverseBRank=rank,
            inverseWRank=rank,
        )
        rows.append(metadata.get_metadata_row(1, 361))
    return torch.as_tensor(
        np.repeat(np.stack(rows), positions, axis=0), device=device
    )


def scores(
    model, tensors: list[torch.Tensor], metadata: torch.Tensor
) -> torch.Tensor:
    """Returns four-game by ten-candidate mean played-move log likelihoods."""
    output = model(
        tensors[0].repeat(10, 1, 1, 1),
        tensors[1].repeat(10, 1),
        input_meta=metadata,
    )
    log_policy = output[0][0][:, 0].log_softmax(1)
    played = log_policy.gather(1, tensors[3].repeat(10)[:, None]).squeeze(1)
    return played.reshape(10, 4, 8).mean(2).transpose(0, 1)


def measure(
    model, tensors: list[torch.Tensor], metadata: torch.Tensor, device: str
) -> dict:
    """Verifies finite candidate-conditioned gradients without saving weights."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            name.startswith(("blocks.16.", "blocks.17.", "norm_trunkfinal."))
        )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, weight_decay=0.01)
    before = trainable[0].detach().clone()
    torch.cuda.reset_peak_memory_stats(device=device)
    started = time.perf_counter()
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        values = scores(model, tensors, metadata)
        loss = torch.nn.functional.cross_entropy(
            values, torch.arange(4, device=device)
        )
        loss.backward()
        if (
            trainable[0].grad is None
            or not torch.isfinite(trainable[0].grad).all()
        ):
            raise ValueError("Candidate-policy gradient missing or nonfinite")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device=device)
    if torch.equal(before, trainable[0]):
        raise ValueError("Candidate-policy objective did not update backbone")
    return {
        "synthetic_losses_not_rank_scores": losses,
        "score_shape": list(values.shape),
        "seconds_for_three_updates": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device=device),
        "physical_positions_per_update": 32,
        "profile_conditioned_positions_per_update": 320,
        "all_ten_profiles_evaluated_per_game": True,
        "true_rank_used_in_metadata": False,
        "backbone_updated": True,
        "weights_saved": False,
    }


def main() -> None:
    """Runs an isolated GPU1 feasibility probe on training positions only."""
    args = common.parse_arguments(
        __doc__ or "", "policy_adaptation_probe_20261109", True
    )
    if args.device != "cuda:1":
        raise ValueError("Candidate policy probe uses GPU1 only")
    if args.output.exists():
        raise FileExistsError(f"Refusing existing policy probe: {args.output}")
    modules, payload = bridge.load_assets(args.research)
    model = bridge.build_model(modules, payload, args.device, False)
    tensors = probe.training_inputs(args, modules, payload)
    metadata = candidate_metadata(modules, len(tensors[3]), args.device)
    result = measure(model, tensors, metadata, args.device)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
