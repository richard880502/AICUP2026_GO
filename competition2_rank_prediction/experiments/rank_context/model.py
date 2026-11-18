"""Low-parameter registered-rank context models for Competition 2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# How HD>=2 evidence is weighted once it is no longer discarded.
# "global"  one learned gate per side, applied to every HD>=2 game;
# "alone"   applied only in questions that have no HD0/HD1 game at all;
# "decay"   divided by 1 + softplus(d) * (ordinary games in the question).
HANDICAP_MODES = ("global", "alone", "decay")


def inverse_softplus(value: float) -> float:
    """Stable inverse of softplus for positive initialization targets."""
    return math.log(math.expm1(value))


class RankContextModel(torch.nn.Module):  # pylint: disable=too-many-instance-attributes
    """v47 opponent identity plus optional handicap/context residual evidence.

    The v47 path is unchanged for HD0/HD1. With use_handicap=True, HD>=2
    games are no longer discarded. Their transition is a low-parameter
    Gaussian over own-rank minus opponent-rank whose center depends on exact
    handicap size and on whether the asked player gives or receives handicap.

    With use_context=True, a zero-initialized linear residual head adds
    question-level structural rank-context logits from handicap/color
    summaries. This head cannot inspect own-move features.
    """

    def __init__(
        self,
        ranks: torch.Tensor,
        context_dimensions: int,
        use_handicap: bool,
        use_context: bool,
        handicap_mode: str = "global",
    ) -> None:
        super().__init__()
        if handicap_mode not in HANDICAP_MODES:
            raise ValueError(f"unknown handicap mode: {handicap_mode}")
        self.ranks = ranks
        self.use_handicap = use_handicap
        self.use_context = use_context
        self.handicap_mode = handicap_mode

        # Exact v47 parameters/initialization.
        self.log_temperature = torch.nn.Parameter(
            torch.tensor(math.log(0.04), device=ranks.device)
        )
        self.unknown = torch.nn.Parameter(
            torch.full((3,), -1.0, device=ranks.device)
        )
        self.prior = torch.nn.Parameter(
            torch.zeros(10, device=ranks.device)
        )
        self.offsets = torch.nn.Parameter(
            torch.zeros(3, 7, device=ranks.device)
        )
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(0.0, device=ranks.device)
        )
        self.gamma = torch.nn.Parameter(
            torch.tensor(0.5, device=ranks.device)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(10, device=ranks.device)
        )
        self.base_weight = torch.nn.Parameter(
            torch.tensor(0.0, device=ranks.device)
        )

        rank_axis = torch.arange(10, device=ranks.device)
        rank_diff = rank_axis[None] - rank_axis[:, None]
        self.register_buffer(
            "offset_index",
            rank_diff.clamp(-3, 3).long() + 3,
        )
        self.register_buffer("rank_diff", rank_diff.float())

        # HD>=2: side 0 = own White / gives handicap, side 1 = own Black /
        # receives handicap. Initialize to roughly one rank per extra stone.
        center_initial = inverse_softplus(1.0)
        slope_initial = inverse_softplus(1.0)
        sigma_initial = inverse_softplus(0.7)  # 0.5 + 0.7 = 1.2
        self.hd_center_raw = torch.nn.Parameter(
            torch.full((2,), center_initial, device=ranks.device)
        )
        self.hd_slope_raw = torch.nn.Parameter(
            torch.full((2,), slope_initial, device=ranks.device)
        )
        self.hd_sigma_raw = torch.nn.Parameter(
            torch.full((2,), sigma_initial, device=ranks.device)
        )
        self.hd_unknown = torch.nn.Parameter(
            torch.full((2,), -1.0, device=ranks.device)
        )
        # Scarcity decay: HD>=2 evidence is wide (learned sigma near 2 ranks),
        # so multiplying it in beside sharp even-game evidence dilutes the
        # latter. Zero init reproduces the "global" mode exactly.
        self.hd_decay_raw = torch.nn.Parameter(
            torch.zeros((), device=ranks.device)
        )
        self.hd_gate_raw = torch.nn.Parameter(
            torch.zeros(2, device=ranks.device)
        )

        if use_context:
            self.context_head = torch.nn.Linear(
                context_dimensions, 10, bias=False, device=ranks.device
            )
            torch.nn.init.zeros_(self.context_head.weight)
        else:
            self.context_head = None

    def _posterior(
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        hd_mask: torch.Tensor,
        side: torch.Tensor,
    ) -> torch.Tensor:
        """Player-ID cosine -> opponent registered-rank posterior."""
        posterior = (
            scores / self.log_temperature.exp()
        ).softmax(1) @ self.ranks
        ordinary_unknown = torch.sigmoid(
            self.unknown[kind.clamp(max=2)]
        )
        if self.use_handicap:
            hd_unknown = torch.sigmoid(self.hd_unknown[side])
            unknown = torch.where(
                hd_mask, hd_unknown, ordinary_unknown
            )
        else:
            unknown = ordinary_unknown
        return (
            (1 - unknown[:, None]) * posterior
            + unknown[:, None] * self.prior.softmax(0)
        )

    def _hd_transition(
        self,
        handicap: torch.Tensor,
        own_black: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """P(own rank | opponent rank, exact handicap, receiver/giver)."""
        side = own_black.long()
        stones_after_two = (handicap.float() - 2.0).clamp(min=0.0)
        center_abs = (
            F.softplus(self.hd_center_raw[side])
            + F.softplus(self.hd_slope_raw[side]) * stones_after_two
        )
        direction = torch.where(
            own_black,
            torch.full_like(center_abs, -1.0),
            torch.ones_like(center_abs),
        )
        center = direction * center_abs
        sigma = 0.5 + F.softplus(self.hd_sigma_raw[side])
        z = (
            self.rank_diff[None] - center[:, None, None]
        ) / sigma[:, None, None]
        transition = (-0.5 * z.square()).log_softmax(2)
        gate = 2.0 * torch.sigmoid(self.hd_gate_raw[side])
        return transition, gate

    def forward(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        scores: torch.Tensor,
        kind: torch.Tensor,
        game_question: torch.Tensor,
        base: torch.Tensor,
        handicap: torch.Tensor,
        own_black: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """Returns question rank logits."""
        questions = len(base)
        hd_mask = handicap >= 2
        side = own_black.long()
        posterior = self._posterior(scores, kind, hd_mask, side)

        ordinary = self.offsets[:, self.offset_index].log_softmax(2)
        transition = ordinary[kind.clamp(max=2)]
        game_weight = (kind < 3).to(base.dtype)

        if self.use_handicap:
            hd_transition, hd_gate = self._hd_transition(
                handicap, own_black
            )
            transition = torch.where(
                hd_mask[:, None, None],
                hd_transition,
                transition,
            )
            hd_weight = hd_gate.to(base.dtype)
            if self.handicap_mode != "global":
                ordinary_games = torch.zeros(
                    questions, device=base.device, dtype=base.dtype
                ).index_add_(
                    0, game_question, (kind < 3).to(base.dtype)
                )[game_question]
                if self.handicap_mode == "alone":
                    # Only speak where nothing else does: questions whose every
                    # game is HD>=2 currently receive no opponent evidence at
                    # all. Every other question keeps exact v47 behaviour.
                    hd_weight = hd_weight * (ordinary_games == 0).to(base.dtype)
                else:
                    hd_weight = hd_weight / (
                        1.0
                        + F.softplus(self.hd_decay_raw) * ordinary_games
                    )
            game_weight = torch.where(
                hd_mask,
                hd_weight,
                torch.ones_like(game_weight),
            )

        per_game = torch.logsumexp(
            posterior.clamp(min=1e-9).log()[:, :, None] + transition,
            dim=1,
        )
        per_game = per_game * game_weight[:, None]
        total = torch.zeros(
            questions, 10, device=base.device
        ).index_add_(0, game_question, per_game)
        used = torch.zeros(
            questions, device=base.device
        ).index_add_(0, game_question, game_weight).clamp(min=1e-6)

        logits = (
            self.log_alpha.exp()
            * total
            / used[:, None] ** self.gamma
            + self.bias
            + torch.sigmoid(self.base_weight)
            * 2.0
            * base.clamp(min=1e-9).log()
        )
        if self.context_head is not None:
            logits = logits + self.context_head(context)
        return logits

    def regularization(self, context_l2: float) -> torch.Tensor:
        """L2 penalty only for the new residual head."""
        if self.context_head is None:
            return self.bias.new_zeros(())
        return context_l2 * self.context_head.weight.square().mean()

    def diagnostic_parameters(self) -> dict:
        """Small interpretable snapshot for experiment reports."""
        output: dict[str, object] = {
            "temperature": float(self.log_temperature.exp().detach().cpu()),
            "base_weight": float(
                (2.0 * torch.sigmoid(self.base_weight)).detach().cpu()
            ),
        }
        if self.use_handicap:
            centers = {}
            for side, label in ((0, "own_gives"), (1, "own_receives")):
                sign = 1.0 if side == 0 else -1.0
                values = {}
                for handicap in (2, 3, 4, 5, 6):
                    magnitude = (
                        F.softplus(self.hd_center_raw[side])
                        + F.softplus(self.hd_slope_raw[side])
                        * (handicap - 2)
                    )
                    values[str(handicap)] = float(
                        (sign * magnitude).detach().cpu()
                    )
                centers[label] = values
            output["hd_center_own_minus_opp"] = centers
            output["hd_sigma"] = [
                float(value)
                for value in (
                    0.5 + F.softplus(self.hd_sigma_raw)
                ).detach().cpu()
            ]
            output["hd_gate"] = [
                float(value)
                for value in (
                    2.0 * torch.sigmoid(self.hd_gate_raw)
                ).detach().cpu()
            ]
            output["hd_unknown"] = [
                float(value)
                for value in torch.sigmoid(
                    self.hd_unknown
                ).detach().cpu()
            ]
            output["handicap_mode"] = self.handicap_mode
            if self.handicap_mode == "decay":
                output["hd_decay"] = float(
                    F.softplus(self.hd_decay_raw).detach().cpu()
                )
        return output
