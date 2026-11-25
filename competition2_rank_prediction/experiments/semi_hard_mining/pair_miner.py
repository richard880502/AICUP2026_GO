"""Selects adjacent training players using detached semi-hard strengths.

The training forward must recompute the returned game bags with gradients.
"""

import numpy as np
import torch

from competition2_rank_prediction.experiments.semi_hard_mining import (
    player_sampler,
)


class SemiHardMiner:
    """Mines unique 1D/2D training pairs in the positive-margin window.

    Attributes:
        data: Player-disjoint feature inventory.
        pools: Eligible dense training players per rank.
        strengths: Detached current-model player strengths.
        ordered_high: High-rank training players sorted by strength.
        high_scores: Corresponding sorted scalar values.
        counts: Cumulative mining and refresh diagnostics.
    """

    def __init__(self, data: player_sampler.PlayerData) -> None:
        self.data = data
        self.pools = {}
        for rank in (4, 5):
            self.pools[rank] = np.asarray(
                [
                    str(player)
                    for player in data.splits["train"]
                    if len(data.rows[str(player)]) >= 20
                    and data.labels[data.rows[str(player)][0]] == rank
                ]
            )
            if len(self.pools[rank]) < 4:
                raise ValueError(
                    "Mining requires four dense training players per rank"
                )
        self.strengths: dict[str, float] = {}
        self.ordered_high = np.asarray([], dtype=str)
        self.high_scores = np.asarray([], dtype=float)
        self.counts = {
            "refreshes": 0,
            "random": 0,
            "mined": 0,
            "fallback": 0,
            "mined_attempts": 0,
            "mined_margin_min": 1.0,
            "mined_margin_max": 0.0,
        }

    def refresh(
        self,
        model: torch.nn.Module,
        tokens: torch.Tensor,
        masks: torch.Tensor,
    ) -> None:
        """Refreshes training strengths without changing weights or model mode."""
        players = np.concatenate(list(self.pools.values()))
        rows = np.concatenate(
            [self.data.rows[str(player)] for player in players]
        )
        scores = np.zeros(len(tokens), dtype=np.float32)
        previous_mode = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for start in range(0, len(rows), 256):
                    indices = rows[start : start + 256]
                    encoded = model.encode(tokens[indices], masks[indices])
                    scores[indices] = (
                        model.score(encoded).flatten().cpu().numpy()
                    )
        finally:
            model.train(previous_mode)
        self.update_strengths(
            {
                str(player): float(scores[self.data.rows[str(player)]].mean())
                for player in players
            }
        )

    def update_strengths(self, strengths: dict[str, float]) -> None:
        """Installs finite training-only strengths and builds a sorted index."""
        expected = set(np.concatenate(list(self.pools.values())))
        if (
            set(strengths) != expected
            or not np.isfinite(list(strengths.values())).all()
        ):
            raise ValueError(
                "Strength cache must cover exactly eligible training players"
            )
        self.strengths = dict(strengths)
        values = np.asarray(
            [strengths[str(player)] for player in self.pools[5]]
        )
        order = np.argsort(values)
        self.ordered_high = self.pools[5][order]
        self.high_scores = values[order]
        self.counts["refreshes"] += 1

    def _mined_players(
        self,
        generator: np.random.Generator,
        used: set[str],
    ) -> tuple[str, str] | None:
        for low in generator.permutation(self.pools[4]):
            if str(low) in used:
                continue
            strength = self.strengths[str(low)]
            left = np.searchsorted(self.high_scores, strength, side="right")
            right = np.searchsorted(
                self.high_scores, strength + 1, side="right"
            )
            available = [
                str(player)
                for player in self.ordered_high[left:right]
                if str(player) not in used
            ]
            if available:
                return str(low), str(generator.choice(available))
        return None

    def _players(
        self,
        generator: np.random.Generator,
        used: set[str],
    ) -> tuple[str, str]:
        if generator.random() < 0.30:
            self.counts["mined_attempts"] += 1
            pair = self._mined_players(generator, used)
            if pair is not None:
                margin = self.strengths[pair[1]] - self.strengths[pair[0]]
                if not 0 < margin <= 1:
                    raise RuntimeError(
                        "Selected mined pair violates the margin window"
                    )
                self.counts["mined"] += 1
                self.counts["mined_margin_min"] = min(
                    self.counts["mined_margin_min"], margin
                )
                self.counts["mined_margin_max"] = max(
                    self.counts["mined_margin_max"], margin
                )
                return pair
            self.counts["fallback"] += 1
        self.counts["random"] += 1
        players = []
        for rank in (4, 5):
            available = [
                player for player in self.pools[rank] if str(player) not in used
            ]
            players.append(str(generator.choice(available)))
        return players[0], players[1]

    def sample_pairs(
        self,
        generator: np.random.Generator,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Returns four ordinal bags with no repeated players in the step."""
        if not self.strengths:
            raise RuntimeError("Refresh training strengths before mining")
        used: set[str] = set()
        pairs = []
        for _ in range(4):
            players = self._players(generator, used)
            used.update(players)
            bags = []
            for player in players:
                count = int(generator.integers(5, 21))
                bags.append(
                    generator.choice(
                        self.data.rows[player], count, replace=False
                    )
                )
            pairs.append((bags[0], bags[1]))
        return pairs
