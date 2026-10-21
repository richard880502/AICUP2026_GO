"""Confusion matrices and UMAP views of the competition-2 features and model representations.

Outputs (experiments/visualization/figures/):
  confusion.png          question-level confusion on the untouched holdout (production best2)
                         + per-game confusion on validation
  umap_raw_features.png  UMAP of per-game raw feature statistics (validation games)
  umap_embeddings.png    UMAP of the dense Transformer's learned game vectors: per game and
                         per player
  umap_by_rank.png       small multiples: player-level embedding, one rank highlighted per panel
Run from the project root:
  c1/.venv/bin/python experiments/visualization/make_figures.py --holdout-pred <submission.csv>
"""
import argparse
import json
import pathlib
import sys
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import umap  # type: ignore[import-untyped]
from scipy.stats import spearmanr  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error
import transformer_rank as tr  # noqa: E402  pylint: disable=wrong-import-position,import-error

OUT = ROOT / "experiments/visualization/figures"
L = list(lib.LEVELS)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#1f1f1d", "#5f5e5a", "#d6d5d0", "#e8e7e3"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
       "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]   # blue 100 -> 700, sequential
GROUP_RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # validated 5-step ramp
GROUPS = ["D-C", "B-A", "1D-2D", "3D-4D", "5D-6D"]


def confusion(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """10x10 count matrix, rows = true rank, columns = predicted rank."""
    m = np.zeros((10, 10), int)
    for p, t in zip(pred, truth):
        m[t, p] += 1
    return m


def draw_confusion(ax: Any, m: np.ndarray, title: str) -> tuple[float, float]:
    """Row-normalised heatmap of a confusion matrix; returns (exact, off-by-one) shares."""
    share = m / np.maximum(m.sum(1, keepdims=True), 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", [SURFACE] + SEQ)
    ax.imshow(share, cmap=cmap, vmin=0, vmax=0.8)
    for i in range(10):
        for j in range(10):
            if m[i, j]:
                ax.text(j, i, f"{100 * share[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if share[i, j] > 0.45 else INK)
    ax.set_xticks(range(10), L)
    ax.set_yticks(range(10), L)
    ax.set_xlabel("predicted rank")
    ax.set_ylabel("true rank")
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    exact = np.trace(m) / m.sum()
    adj = sum(m[i, i + 1] + m[i + 1, i] for i in range(9)) / m.sum()
    return exact, adj


def game_vectors(model: Any, x: np.ndarray, valid: np.ndarray, device: torch.device,
                 batch: int = 1024) -> np.ndarray:
    """The attention-pooled 128-d game representation (input of the ordinal head)."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model.encode(torch.from_numpy(x[i:i + batch]).to(device),
                                    torch.from_numpy(valid[i:i + batch]).to(device)).cpu().numpy())
    return np.concatenate(out)


def scatter_groups(ax: Any, xy: np.ndarray, y: np.ndarray, title: str,
                   marker: tuple[float, float] = (3, 0.6)) -> None:
    """Scatter of 2-D points coloured by the five rank groups; marker = (size, alpha)."""
    size, alpha = marker
    g = np.asarray(y) // 2
    for k in range(5):  # light groups first so the darker (stronger) groups sit on top
        s = g == k
        ax.scatter(xy[s, 0], xy[s, 1], s=size, c=GROUP_RAMP[k], alpha=alpha, linewidths=0,
                   label=GROUPS[k])
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def confusion_figure(holdout_pred: pathlib.Path) -> dict[str, Any]:
    """confusion.png and the holdout / validation-game confusion summary."""
    sub = pd.read_csv(holdout_pred).set_index("question_id")
    lab = pd.read_csv(ROOT / "features/holdout/tests/labels.csv")
    qt = np.array([L.index(r) for r in lab["rank"]])
    qp = np.array([L.index(sub.loc[q, "pred_rank"]) for q in lab["question_id"]])
    v = np.load(ROOT / "results/val_probs_d96.npz", allow_pickle=True)
    gp, gt = v["transformer"].argmax(1), v["y"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    e1, a1 = draw_confusion(axes[0], confusion(qp, qt),
                            "Holdout questions (400, production best2)\nrow % of each true rank")
    e2, a2 = draw_confusion(axes[1], confusion(gp, gt), "Validation single games "
                            "(8,000, dense Transformer)\nrow % of each true rank")
    fig.text(0.01, 0.01,
             f"Holdout: exact {e1:.1%}, off-by-one {a1:.1%}, worse {1 - e1 - a1:.1%}.   "
             f"Single games: exact {e2:.1%}, off-by-one {a2:.1%}, worse {1 - e2 - a2:.1%}.",
             color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "confusion.png", dpi=150)
    m = confusion(qp, qt)
    return {
        "holdout_confusion": m.tolist(),
        "holdout_per_rank": {
            L[i]: {"n": int(m[i].sum()), "exact": float(m[i, i] / m[i].sum()),
                   "off_by_one": float(sum(m[i, j] for j in (i - 1, i + 1) if 0 <= j < 10)
                                       / m[i].sum()),
                   "mean_signed_error": float(np.sum((np.arange(10) - i) * m[i]) / m[i].sum())}
            for i in range(10)},
        "holdout_totals": {"exact": e1, "off_by_one": a1},
        "val_game_totals": {"exact": e2, "off_by_one": a2},
    }


def raw_feature_umap(val: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """UMAP of the standardised per-game mean / std of every dense feature."""
    mask = val["masks"][..., None]
    cnt = np.maximum(mask.sum(1), 1)
    mean = (x * mask).sum(1) / cnt
    std = np.sqrt(((x - mean[:, None]) ** 2 * mask).sum(1) / cnt)
    raw = np.concatenate([mean, std], 1)
    raw = (raw - raw.mean(0)) / (raw.std(0) + 1e-6)
    return umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(raw)


def load_dense_model(x: np.ndarray, dev: torch.device) -> tuple[Any, dict[str, Any]]:
    """The production dense Transformer (seed 0) and its checkpoint."""
    ck = torch.load(ROOT / "results/d96_seed0.pt", map_location="cpu", weights_only=False)
    ar = ck["args"]
    model = tr.RankTransformer(x.shape[2], x.shape[1], ar["d_model"], ar["heads"], ar["ffn"],
                               ar["layers"], ar["dropout"], ar.get("head", "clm")).to(dev)
    model.load_state_dict(ck["state"])
    return model, ck


def learned_representation(val: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray,
                           dev: torch.device) -> tuple[Any, tuple[np.ndarray, np.ndarray],
                                                       tuple[np.ndarray, np.ndarray]]:
    """Seed-0 dense Transformer vectors of the validation games: (model, (player-mean vectors,
    player ranks), (UMAP of the game vectors, UMAP of the player vectors))."""
    model, ck = load_dense_model(x, dev)
    xs = np.where(val["masks"][..., None], (x - ck["mean"]) / ck["std"], 0).astype(np.float32)
    emb = game_vectors(model, xs, val["masks"], dev)
    xy_emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(emb)
    players = np.unique(val["player_id"])
    pemb = np.stack([emb[val["player_id"] == p].mean(0) for p in players])
    py = np.array([y[val["player_id"] == p][0] for p in players])
    xy_p = umap.UMAP(n_neighbors=15, min_dist=0.2, random_state=0).fit_transform(pemb)
    return model, (pemb, py), (xy_emb, xy_p)


def save_umap_figures(xy_raw: np.ndarray, y: np.ndarray, xy_emb: np.ndarray,
                      players: tuple[np.ndarray, np.ndarray]) -> None:
    """umap_raw_features.png, umap_embeddings.png and umap_by_rank.png;
    players = (player-level UMAP coordinates, player ranks)."""
    xy_p, py = players
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    scatter_groups(ax, xy_raw, y, "UMAP of raw per-game KataGo/HumanSL statistics\n"
                   "(8,000 validation games, colour = rank group)")
    ax.legend(title="rank group", frameon=False, markerscale=4, loc="best", fontsize=9,
              title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "umap_raw_features.png", dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    scatter_groups(axes[0], xy_emb, y,
                   "Learned game vectors (dense Transformer)\n8,000 validation games")
    scatter_groups(axes[1], xy_p, py,
                   "Learned vectors averaged per player\n400 validation players x 20 games",
                   marker=(22, 0.9))
    axes[1].legend(title="rank group", frameon=False, markerscale=1.2, loc="best", fontsize=9,
                   title_fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "umap_embeddings.png", dpi=150)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    for k, ax in enumerate(axes.ravel()):
        ax.scatter(xy_p[:, 0], xy_p[:, 1], s=10, c=MUTED, linewidths=0)
        s = py == k
        ax.scatter(xy_p[s, 0], xy_p[s, 1], s=22, c="#256abf", edgecolors=SURFACE, linewidths=0.8)
        ax.set_title(L[k], fontsize=11, color=INK, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Player-level learned vectors: each panel highlights one true rank "
                 "(40 players each)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "umap_by_rank.png", dpi=150)


def main() -> None:
    """Writes the figures and figures/summary.json."""
    matplotlib.use("Agg")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": MUTED, "font.size": 10})
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-pred", type=pathlib.Path, required=True,
                    help="predict_test.py output on features/holdout/tests")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    # ---- 1. confusion matrices
    summary = confusion_figure(a.holdout_pred)

    # ---- 2. UMAP of raw per-game feature statistics (validation)
    val = lib.load_features(ROOT / "features/feat96/val")
    x = tr.decision_tokens(val["tokens"])
    xy_raw = raw_feature_umap(val, x)
    y = lib.rank_ids(val["rank"])

    # ---- 3. learned representation of the production dense Transformer (seed 0), same games
    model, (pemb, py), (xy_emb, xy_p) = learned_representation(val, x, y, dev)
    save_umap_figures(xy_raw, y, xy_emb, (xy_p, py))

    # how well does a 1-D position along the learned space track rank?
    # (Spearman of UMAP-free strength score)
    with torch.no_grad():
        score = model.score(torch.from_numpy(pemb).to(dev)).cpu().numpy().ravel()
    summary["player_strength_score_spearman_vs_rank"] = float(spearmanr(score, py).correlation)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "holdout_confusion"}, indent=1))


if __name__ == "__main__":
    main()
