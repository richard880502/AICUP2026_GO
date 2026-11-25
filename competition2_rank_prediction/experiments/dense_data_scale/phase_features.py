"""Summarizes decision-distribution shape for matched nonsequence probes.

Call summarize(tokens, masks, rich=True) on cached four-profile decisions.
The control retains mean/std only with identical phase boundaries/channels.
"""

import numpy as np


def phase_statistics(
    values: np.ndarray, masks: np.ndarray, rich: bool
) -> np.ndarray:
    """Computes masked moments and optional quantiles/skew without NaN output."""
    count = np.maximum(masks.sum(axis=1, keepdims=True), 1)
    observed = np.where(masks[..., None], values, 0)
    mean = observed.sum(axis=1) / count
    centered = np.where(masks[..., None], values - mean[:, None], 0)
    variance = (centered**2).sum(axis=1) / count
    std = np.sqrt(variance)
    parts = [mean, std]
    if rich:
        quantile_input = np.where(masks[..., None], values, np.nan)
        empty = ~masks.any(axis=1)
        quantile_input[empty, 0] = 0
        quantiles = np.nanquantile(
            quantile_input, (0.1, 0.25, 0.5, 0.75, 0.9), axis=1
        )
        parts.extend(quantiles)
        skew = (centered**3).sum(axis=1) / count / np.maximum(std, 1e-6) ** 3
        parts.append(np.clip(skew, -10, 10))
    return np.concatenate(parts, axis=1).astype(np.float32)


def summarize(tokens: np.ndarray, masks: np.ndarray, rich: bool) -> np.ndarray:
    """Uses fixed 1/2/4 phases and identical raw/profile-centered channels."""
    if tokens.ndim != 4 or tokens.shape[2:] != (4, 3):
        raise ValueError(
            "Expected games x decisions x four profiles x three features"
        )
    if masks.shape != tokens.shape[:2] or tokens.shape[1] < 4:
        raise ValueError("Invalid decision mask or fewer than four decisions")
    if not np.isfinite(tokens).all():
        raise ValueError("Nonfinite input decision features")
    human = tokens[:, :, 1:, 0]
    centered = human - human.mean(axis=2, keepdims=True)
    values = np.concatenate(
        (tokens.reshape(*tokens.shape[:2], 12), centered), axis=2
    )
    parts = []
    for phases in (1, 2, 4):
        boundaries = np.linspace(0, tokens.shape[1], phases + 1, dtype=int)
        for phase in range(phases):
            start, end = boundaries[phase : phase + 2]
            parts.append(
                phase_statistics(
                    values[:, start:end], masks[:, start:end], rich
                )
            )
    return np.concatenate(parts, axis=1)
