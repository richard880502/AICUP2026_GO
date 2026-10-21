"""Why are A and 1D (and 1D/2D, 4D/5D) hard to separate?  Three experiments on the same simulated
questions as experiments/strength_axis (validation cohort, 20 questions per player, 5-20 games),
evaluated per adjacent boundary.

Exp 1  relative move probability between adjacent HumanSL ranks,
       d_t = log P(m | 1d) - log P(m | 2k) (A = 1-3 kyu -> rank_2k),
       from the 32-decision 10-profile features (features/feat). Also 5k->2k and 1d->2d for context.
Exp 2  error profile: quantiles / extreme-error shares / phase means of the normal-model surprisal
       (96 decisions, features/feat96) and of the score-loss / win-rate-loss per move
       (32 decisions, features/feat_value).
Exp 3  label reliability with what the data allows (no ratings, dates or results exist in the
       CSV/SGF): games per player by rank, model-estimated opponent strength (opponent-side
       features of 2,000 validation games), within-rank spread.

All classifiers are fitted with 5-fold player-disjoint CV inside each boundary (validation cohort
only; the holdout is not used here because its 32-decision value features are incomplete and it
has only 40 players per rank).
Run from the project root: c1/.venv/bin/python experiments/a1d_boundary/run.py
"""
import json
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import rankdata  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# The project modules live in ROOT, which is only on sys.path after the insert above.
# `experiments` must be imported before rank_lib, which puts c1/ (with its own `experiments`
# package) in front of sys.path. strength score: same model / standardisation as strength_axis.
from experiments.strength_axis import run as axis  # noqa: E402  pylint: disable=wrong-import-position,import-error
import rank_lib as lib  # noqa: E402  pylint: disable=wrong-import-position,import-error
import residual_shift  # noqa: E402  pylint: disable=wrong-import-position,import-error

OUT = ROOT / "experiments/a1d_boundary/results"
L = list(lib.LEVELS)
BOUNDARIES = [(3, 4), (4, 5), (7, 8)]   # A/1D, 1D/2D, 4D/5D
SURP = 0  # index of surprisal in the last axis
# model order of features/feat (checked against the shard's `profiles` array)
PROF = {name: i for i, name in enumerate(["normal", "rank_11k", "rank_8k", "rank_5k", "rank_2k",
                                          "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d",
                                          "rank_6d"])}
CACHE = OUT / "question_features.csv"
BASELINE = "s (production strength)"
FOLDS, REPEATS = 5, 5


def strength(tokens: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Strength score of every d96_h seed, [seeds, G]."""
    return axis.strength_outputs(tokens, masks)[0]


# ------------------------------------------------------------------ per-question feature builders
def aligned(dirname: str, keys: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Features of another extraction and, for each key, its row there (-1 when missing)."""
    d = lib.load_features(ROOT / dirname)
    idx = {k: i for i, k in enumerate(d["game_key"])}
    rows = np.array([idx.get(k, -1) for k in keys])
    return d, rows


def relative_probability(tokens: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    """Exp 1 features of one question from its features/feat tokens [games, 32, profiles, f]."""
    flat = tokens[masks]
    d_a1d = flat[:, PROF["rank_2k"], SURP] - flat[:, PROF["rank_1d"], SURP]  # log P(1d) - log P(2k)
    d_ba = flat[:, PROF["rank_5k"], SURP] - flat[:, PROF["rank_2k"], SURP]
    d_12 = flat[:, PROF["rank_1d"], SURP] - flat[:, PROF["rank_2d"], SURP]
    return {"d_1d_vs_2k_mean": d_a1d.mean(), "d_1d_vs_2k_share_pos": (d_a1d > 0).mean(),
            "d_1d_vs_2k_q90": np.quantile(d_a1d, 0.9), "d_2k_vs_5k_mean": d_ba.mean(),
            "d_2d_vs_1d_mean": d_12.mean()}


def surprisal_profile(t: np.ndarray, m: np.ndarray) -> dict[str, Any]:
    """Exp 2 surprisal features of one question from its feat96 tokens and masks."""
    n = m.sum(1, keepdims=True)
    rel = np.arange(t.shape[1])[None] / np.maximum(n, 1)
    ns = t[:, :, 0, SURP]
    v = ns[m]
    f = {"surp_mean": v.mean(), "surp_q50": np.quantile(v, .5), "surp_q90": np.quantile(v, .9),
         "surp_q98": np.quantile(v, .98), "surp_share_gt6": (v > 6).mean(),
         "surp_share_gt8": (v > 8).mean(), "surp_std": v.std()}
    for name, lo, hi in (("opening", 0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("endgame", 2 / 3, 1.01)):
        sel = m & (rel >= lo) & (rel < hi)
        f[f"surp_{name}_mean"] = ns[sel].mean() if sel.any() else np.nan
        f[f"surp_{name}_q90"] = np.quantile(ns[sel], .9) if sel.any() else np.nan
    return f


def error_profile(dense: tuple[np.ndarray, np.ndarray], value: tuple[np.ndarray, np.ndarray],
                  vnames: list[str]) -> dict[str, Any]:
    """Exp 2 features of one question: dense = feat96 (tokens, masks), value = feat_value
    (tokens, masks) with feature names vnames."""
    f = surprisal_profile(*dense)
    tv, mv = value
    sl = tv[..., vnames.index("score_lead_drop_scaled")][mv] * 20.0   # points lost by the move
    wr = tv[..., vnames.index("winrate_drop")][mv]
    f.update({"score_loss_mean": sl.mean(), "score_loss_q90": np.quantile(sl, .9),
              "score_loss_share_gt5": (sl > 5).mean(), "score_loss_share_gt10": (sl > 10).mean(),
              "winrate_loss_mean": wr.mean(), "winrate_loss_share_gt10pc": (wr > .1).mean(),
              "winrate_loss_share_gt25pc": (wr > .25).mean()})
    return f


def extra_columns(table: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Names of the Exp 1 and Exp 2 feature columns."""
    exp1 = [c for c in table.columns if c.startswith("d_")]
    exp2 = [c for c in table.columns if c.startswith(("surp_", "score_loss", "winrate_loss"))]
    return exp1, exp2


# ------------------------------------------------------------------ evaluation per boundary
def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC (ties averaged); ~100x faster than sklearn for the bootstrap."""
    r = rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def stratified_folds(lab: pd.Series, r: np.random.Generator) -> dict[Any, int]:
    """Player -> fold, players of each label dealt round-robin in random order."""
    fold_of: dict[Any, int] = {}
    for label in (0, 1):
        ps = r.permutation(lab.index[lab.values == label].values)
        fold_of.update(zip(ps, np.arange(len(ps)) % FOLDS))
    return fold_of


def cv_auc(df: pd.DataFrame, cols: list[str], c: float = 0.5, max_iter: int = 2000) -> np.ndarray:
    """Out-of-fold probabilities, player-disjoint folds STRATIFIED by the player's label, averaged
    over repeats. (Unstratified folds give each test fold a training intercept that is
    anti-correlated with its own label mix, which pushes the pooled AUC of weak features below
    0.5 - the first run of this script had that bug.)"""
    lab = df.groupby("player")["hi"].first()
    x = df[cols].values
    oof = np.zeros(len(df))
    for rep in range(REPEATS):
        f_ = df["player"].map(stratified_folds(lab, np.random.default_rng(rep))).values
        for k in range(FOLDS):
            tr_, te_ = f_ != k, f_ == k
            sc = StandardScaler().fit(x[tr_])
            clf = LogisticRegression(C=c, max_iter=max_iter).fit(sc.transform(x[tr_]),
                                                                 df["hi"].values[tr_])
            oof[te_] += clf.decision_function(sc.transform(x[te_])) / REPEATS
    return oof


def player_boot_auc_diff(df: pd.DataFrame, a: np.ndarray, b: np.ndarray, reps: int = 2000,
                         seed: int = 1) -> tuple[float, float, float]:
    """Player bootstrap of AUC(a) - AUC(b): mean and 95% interval."""
    r = np.random.default_rng(seed)
    ps = np.array(sorted(df["player"].unique()))
    idx = {p: np.flatnonzero(df["player"].values == p) for p in ps}
    d = []
    for _ in range(reps):
        pick = np.concatenate([idx[p] for p in r.choice(ps, len(ps))])
        yy = df["hi"].values[pick]
        if yy.min() == yy.max():
            continue
        d.append(fast_auc(yy, a[pick]) - fast_auc(yy, b[pick]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def boundary_frame(table: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
    """The questions of two adjacent ranks with hi = 1 for the higher one."""
    df = table[table["y"].isin([lo, hi])].reset_index(drop=True).copy()
    df["hi"] = (df["y"] == hi).astype(int)
    return df


def evaluate_boundary(table: pd.DataFrame, lo: int, hi: int) -> dict[str, Any]:
    """Single-feature AUCs and CV AUCs of every feature set (vs s alone) at one boundary."""
    exp1, exp2 = extra_columns(table)
    sets = {BASELINE: ["s"], "exp1 relative probability": exp1, "exp2 error profile": exp2,
            "s + exp1": ["s"] + exp1, "s + exp2": ["s"] + exp2,
            "s + exp1 + exp2": ["s"] + exp1 + exp2}
    df = boundary_frame(table, lo, hi)
    single = {c: float(roc_auc_score(df["hi"], df[c])) for c in ["s"] + exp1 + exp2}
    oofs = {k: cv_auc(df, cols) for k, cols in sets.items()}
    aucs = {k: float(roc_auc_score(df["hi"], v)) for k, v in oofs.items()}
    diffs = {k: player_boot_auc_diff(df, oofs[k], oofs[BASELINE]) for k in sets if k != BASELINE}
    print(f"\n=== boundary {L[lo]}/{L[hi]}: {len(df)} validation questions from "
          f"{df['player'].nunique()} players")
    print("  single-feature AUC (>0.5 = higher value means the higher rank):")
    for c in ["s"] + exp1 + exp2:
        print(f"    {c:28s} {single[c]:.3f}")
    print("  5-fold player-disjoint CV, logistic regression:")
    for k in sets:
        extra = ("" if k == BASELINE else
                 f"   vs s {diffs[k][0]:+.3f} [{diffs[k][1]:+.3f}, {diffs[k][2]:+.3f}]")
        print(f"    {k:28s} AUC {aucs[k]:.3f}{extra}")
    return {"n_questions": int(len(df)), "n_players": int(df["player"].nunique()),
            "single_feature_auc": single, "cv_auc": aucs, "cv_auc_minus_s": diffs}


# ----------------------------------------------------------------- Exp 3: label reliability proxies
def games_per_player() -> dict[str, dict[str, Any]]:
    """Players and games per player of every rank in the training CSVs."""
    counts = {}
    for rank in L:
        pc = pd.read_csv(ROOT / f"data/training/train_{rank}.csv",
                         usecols=["player_id"])["player_id"].value_counts()
        counts[rank] = {"players": int(len(pc)), "games_per_player_median": float(pc.median()),
                        "games_per_player_mean": float(pc.mean()),
                        "share_players_lt20_games": float((pc < 20).mean())}
    print("  players / median games per player / share of players with < 20 games:")
    for rank in L:
        c = counts[rank]
        print(f"    {rank:3s} {c['players']:5d}   {c['games_per_player_median']:6.0f}   "
              f"{c['share_players_lt20_games']:.2f}")
    return counts


def boundary_aucs(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Single-game AUC of a score for the player's label at every boundary."""
    return {f"{L[lo]}/{L[hi]}": float(roc_auc_score((y[np.isin(y, [lo, hi])] == hi).astype(int),
                                                    score[np.isin(y, [lo, hi])]))
            for lo, hi in BOUNDARIES}


def opponent_strength(opp: dict[str, np.ndarray], val96: dict[str, np.ndarray],
                      standardised: tuple[np.ndarray, np.ndarray, np.ndarray],
                      y_val: np.ndarray) -> dict[str, Any]:
    """Model-estimated opponent strength on the opponent-side validation games;
    standardised = (s_val, per-seed mean, per-seed std)."""
    idx96 = {k: i for i, k in enumerate(val96["game_key"])}
    orow = np.array([idx96[k] for k in opp["game_key"]])
    s_opp = ((strength(opp["tokens"], opp["masks"]) - standardised[1]) / standardised[2]).mean(0)
    s_me = standardised[0][orow]
    yo = y_val[orow]
    corr = np.corrcoef(s_me, s_opp)[0, 1]
    print(f"  opponent strength (model-estimated) on {len(opp['game_key'])} validation games:")
    print(f"    corr(player strength, opponent strength) per game = {corr:.3f}")
    opp_by_rank = {L[r]: {"n": int((yo == r).sum()),
                          "player_median": float(np.median(s_me[yo == r])),
                          "opponent_median": float(np.median(s_opp[yo == r]))} for r in range(10)}
    for rank, v in opp_by_rank.items():
        print(f"    {rank:3s} n={v['n']:4d}  player strength median {v['player_median']:+.2f}   "
              f"opponent strength median {v['opponent_median']:+.2f}")
    opp_auc = boundary_aucs(yo, s_opp)
    own_auc = boundary_aucs(yo, s_me)
    print(f"    single-game AUC of the OPPONENT's strength for the player's label: {opp_auc}")
    print(f"    single-game AUC of the player's OWN strength (same games):         {own_auc}")
    return {"opponent_by_rank": opp_by_rank, "opponent_auc": opp_auc,
            "own_auc_same_games": own_auc, "corr_player_opponent": float(corr)}


# ------------------------------------------------------------------ data
def validation_strength(
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """feat96 validation features and (s_val, per-seed mean, per-seed std) of the strength score."""
    val96 = lib.load_features(ROOT / "features/feat96/val")
    s_raw = strength(val96["tokens"], val96["masks"])
    mu, sd = s_raw.mean(1, keepdims=True), s_raw.std(1, keepdims=True)
    return val96, (((s_raw - mu) / sd).mean(0), mu, sd)


def checked_questions(val96: dict[str, np.ndarray], s_val: np.ndarray,
                      y_val: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """The same questions as experiments/strength_axis/run.py, checked against its saved arrays:
    game indices, label and mean strength of every question."""
    games, q_y = residual_shift.validation_questions(val96["player_id"], y_val)
    saved = np.load(ROOT / "experiments/strength_axis/results/questions.npz", allow_pickle=True)
    q_player = val96["player_id"][[g[0] for g in games]]
    assert (np.array([len(g) for g in games]) == saved["q_k"]).all()
    assert (q_player == saved["q_player"]).all()
    q_s = np.array([s_val[g].mean() for g in games])
    assert np.allclose(q_s, saved["q_s"], atol=1e-4), "question strength differs from strength_axis"
    return games, q_y, q_s


def question_table(val96: dict[str, np.ndarray], games: list[np.ndarray], s_val: np.ndarray,
                   y_val: np.ndarray) -> pd.DataFrame:
    """Per-question s, Exp 1 and Exp 2 features (cached in results/question_features.csv)."""
    old, old_rows = aligned("features/feat/val", val96["game_key"])            # 32 dec. x 11 models
    val_v, val_rows = aligned("features/feat_value/val", val96["game_key"])     # 32 dec. x 32 value
    vnames = list(np.load(ROOT / "features/feat_value/val/feature_names.npy"))
    assert (old_rows >= 0).all() and (val_rows >= 0).all()
    rows = []
    for gi in [] if CACHE.exists() else games:
        f = {"player": val96["player_id"][gi[0]], "y": y_val[gi[0]], "k": len(gi),
             "s": s_val[gi].mean()}
        f.update(relative_probability(old["tokens"][old_rows[gi]], old["masks"][old_rows[gi]]))
        f.update(error_profile((val96["tokens"][gi], val96["masks"][gi]),
                               (val_v["tokens"][val_rows[gi]], val_v["masks"][val_rows[gi]]),
                               vnames))
        rows.append(f)
    table = pd.read_csv(CACHE, dtype={"player": str}) if CACHE.exists() else pd.DataFrame(rows)
    table.to_csv(CACHE, index=False)
    return table


def main() -> None:
    """Runs Exp 1-3 and writes results/report.json."""
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {}
    val96, standardised = validation_strength()
    s_val = standardised[0]
    y_val = lib.rank_ids(val96["rank"])
    games, q_y, q_s = checked_questions(val96, s_val, y_val)
    table = question_table(val96, games, s_val, y_val)
    for lo, hi in BOUNDARIES:
        report[f"{L[lo]}/{L[hi]}"] = evaluate_boundary(table, lo, hi)

    print("\n=== Exp 3: label reliability (data has no ratings / dates / results; "
          "internal proxies only)")
    counts = games_per_player()
    spread = {L[r]: [float(np.quantile(q_s[q_y == r], q)) for q in (.1, .25, .5, .75, .9)]
              for r in range(10)}
    print("  within-rank spread of question strength (IQR) :",
          {k: round(v[3] - v[1], 2) for k, v in spread.items()})

    opp_dir = ROOT / "experiments/opponent_side/features/feat96opp/val"
    # the first opponent extraction re-extracted the PLAYER's moves (bug); wait for the fixed one
    if len(list(opp_dir.glob("feat_*.npz"))) < 2:
        (OUT / "report.json").write_text(json.dumps(report, indent=2))
        print("\nopponent features not ready; saved", OUT / "report.json")
        sys.exit(0)
    opp = opponent_strength(lib.load_features(opp_dir), val96, standardised, y_val)
    report["exp3"] = {"games_per_player": counts, "within_rank_spread": spread,
                      "opponent_by_rank": opp["opponent_by_rank"],
                      "opponent_auc": opp["opponent_auc"],
                      "own_auc_same_games": opp["own_auc_same_games"],
                      "corr_player_opponent": opp["corr_player_opponent"]}
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print("\nsaved", OUT / "report.json")


if __name__ == "__main__":
    main()
