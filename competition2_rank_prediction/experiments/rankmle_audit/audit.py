"""Premise audit: does djma/rankmle's many-profile likelihood add rank information?

rankmle (https://github.com/djma/rankmle) queries KataGo's HumanSL network under
many rank profiles per move and predicts the profile with the highest mean
log-likelihood of the played moves. Production Competition 2 queries only three
HumanSL profiles (rank_5k, rank_1d, rank_5d) and learns an ordinal model on
top. The only ingredient rankmle has that we lack is the extra profiles, so this
audit isolates exactly that.

On the 400-player validation cohort (20 games each, 40 players per rank) with
all ten extractor profiles, it compares:

* ``rankmle``           the faithful rule: per question, sum each game's mean
                        log-likelihood under every profile and take the argmax.
                        Untrained.
* ``rankmle_centered``  the same after subtracting each profile's mean
                        log-likelihood on the training folds (ten parameters),
                        to separate the rule from a constant profile bias.
* ``three``             a fixed learner on the production profiles.
* ``ten``               the same learner on all ten profiles.
* ``ten_loglik``        the same learner on the surprisal channel only, all
                        eleven streams - the information rankmle itself uses.

The default ten profiles map one-to-one onto the ten classes
(11k, 8k, 5k, 2k -> D, C, B, A; 1d..6d -> 1D..6D), so the argmax is directly a
class prediction.

Gate: ``ten - three >= +0.003`` with the paired player-bootstrap 95% lower
bound above zero. Otherwise more profiles do not add information to this
pipeline and the full-corpus extraction is not worth running.

CPU only; reads cached features and trains no network.
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

CLASSES = ("D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D")
TEN = (
    "rank_11k", "rank_8k", "rank_5k", "rank_2k",
    "rank_1d", "rank_2d", "rank_3d", "rank_4d", "rank_5d", "rank_6d",
)
THREE = ("normal", "rank_5k", "rank_1d", "rank_5d")
GAIN = np.array(
    [[1.0 if a == b else (np.exp(-1.0) if abs(a - b) == 1 else 0.0)
      for b in range(10)] for a in range(10)]
)
SEED = 20261118


def load(directory: pathlib.Path) -> dict:
    """Per-game mean of each channel under each stream, plus labels."""
    means: list[np.ndarray] = []
    players: list[str] = []
    labels: list[int] = []
    names: tuple[str, ...] | None = None
    for path in sorted(glob.glob(str(directory / "feat_*.npz"))):
        # NpzFile values are arrays; pylint infers the np.array function.
        # pylint: disable=not-an-iterable
        with np.load(path, allow_pickle=True) as shard:
            tokens = np.asarray(shard["tokens"], dtype=np.float64)
            masks = np.asarray(shard["masks"], dtype=bool)
            stream_names = tuple(str(name) for name in shard["profiles"])
            if names is None:
                names = stream_names
            elif names != stream_names:
                raise ValueError(f"{path}: profile order differs")
            players.extend(str(player) for player in shard["player_id"])
            labels.extend(CLASSES.index(str(rank)) for rank in shard["rank"])
        # pylint: enable=not-an-iterable
        weight = masks[:, :, None, None].astype(np.float64)
        count = weight.sum(1).clip(min=1.0)
        means.append((tokens * weight).sum(1) / count)
    if names is None:
        raise ValueError(f"no feat_*.npz under {directory}")
    missing = [name for name in TEN + THREE if name not in names]
    if missing:
        raise ValueError(f"profiles missing from features: {missing}")
    return {
        "game_means": np.concatenate(means),  # [games, streams, 3]
        "players": np.asarray(players),
        "labels": np.asarray(labels, dtype=np.int64),
        "streams": names,
    }


def make_questions(data: dict, per_player: int) -> dict:
    """Simulated 5-20 game questions, ``per_player`` per player, fixed seed."""
    generator = np.random.default_rng(SEED)
    members, question_player, question_label = [], [], []
    for player in np.unique(data["players"]):
        games = np.flatnonzero(data["players"] == player)
        label = int(data["labels"][games[0]])
        for _ in range(per_player):
            size = int(generator.integers(5, min(20, len(games)) + 1))
            members.append(generator.choice(games, size=size, replace=False))
            question_player.append(player)
            question_label.append(label)
    return {
        "members": members,
        "players": np.asarray(question_player),
        "labels": np.asarray(question_label, dtype=np.int64),
    }


def question_features(data: dict, questions: dict, streams: tuple, channels: tuple) -> np.ndarray:
    """Mean over a question's games of the selected stream/channel means."""
    index = [data["streams"].index(stream) for stream in streams]
    per_game = data["game_means"][:, index][:, :, list(channels)].reshape(
        len(data["game_means"]), -1
    )
    return np.stack([per_game[member].mean(0) for member in questions["members"]])


def profile_loglik(data: dict, questions: dict) -> np.ndarray:
    """Question-level summed mean log-likelihood under each of the ten profiles."""
    index = [data["streams"].index(stream) for stream in TEN]
    loglik = -data["game_means"][:, index, 0]  # channel 0 is surprisal
    return np.stack([loglik[member].sum(0) for member in questions["members"]])


def player_folds(questions: dict, folds: int) -> np.ndarray:
    """Rank-stratified, player-disjoint fold id for every question."""
    generator = np.random.default_rng(SEED + 1)
    fold_of_player = {}
    for label in range(10):
        players = np.unique(questions["players"][questions["labels"] == label])
        generator.shuffle(players)
        for position, player in enumerate(players):
            fold_of_player[player] = position % folds
    return np.asarray([fold_of_player[player] for player in questions["players"]])


def decode(probabilities: np.ndarray) -> np.ndarray:
    """Expected competition-utility decoding."""
    return (probabilities @ GAIN).argmax(1)


def utility(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-question competition utility."""
    return GAIN[prediction, labels]


def learned(features: np.ndarray, labels: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Cross-fitted multinomial logistic probabilities."""
    probabilities = np.zeros((len(labels), 10))
    for held in np.unique(fold):
        train, test = fold != held, fold == held
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(max_iter=4000, C=1.0)
        model.fit(scaler.transform(features[train]), labels[train])
        probabilities[test] = model.predict_proba(scaler.transform(features[test]))
    return probabilities


def centered_rankmle(loglik: np.ndarray, sizes: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Argmax after removing each profile's per-game mean log-likelihood bias."""
    prediction = np.zeros(len(loglik), dtype=np.int64)
    per_game = loglik / sizes[:, None]
    for held in np.unique(fold):
        bias = per_game[fold != held].mean(0)
        test = fold == held
        prediction[test] = (per_game[test] - bias).argmax(1)
    return prediction


def paired(new: np.ndarray, old: np.ndarray, players: np.ndarray, draws: int = 5000) -> dict:
    """Player-bootstrap interval of the mean per-question utility difference."""
    unique, inverse = np.unique(players, return_inverse=True)
    per_player = np.bincount(inverse, weights=new - old) / np.bincount(inverse)
    generator = np.random.default_rng(SEED + 2)
    picks = generator.integers(0, len(unique), size=(draws, len(unique)))
    means = per_player[picks].mean(1)
    return {
        "delta": round(float((new - old).mean()), 6),
        "ci95": [round(float(np.percentile(means, 2.5)), 6),
                 round(float(np.percentile(means, 97.5)), 6)],
    }


def summary(prediction: np.ndarray, labels: np.ndarray) -> dict:
    """Utility, exact rate and gross-error rate."""
    error = np.abs(prediction - labels)
    return {
        "utility": round(float(utility(prediction, labels).mean()), 6),
        "exact": round(float((error == 0).mean()), 4),
        "off_by_two_or_more": round(float((error >= 2).mean()), 4),
    }


def main() -> None:  # pylint: disable=too-many-locals
    """Runs every arm and prints one JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=pathlib.Path, required=True)
    parser.add_argument("--per-player", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    data = load(args.features)
    questions = make_questions(data, args.per_player)
    labels, players = questions["labels"], questions["players"]
    fold = player_folds(questions, args.folds)
    sizes = np.asarray([len(member) for member in questions["members"]], dtype=np.float64)

    loglik = profile_loglik(data, questions)
    predictions = {
        "rankmle": loglik.argmax(1),
        "rankmle_centered": centered_rankmle(loglik, sizes, fold),
    }
    probabilities = {
        "three": learned(question_features(data, questions, THREE, (0, 1, 2)), labels, fold),
        "ten": learned(
            question_features(data, questions, ("normal",) + TEN, (0, 1, 2)), labels, fold
        ),
        "ten_loglik": learned(
            question_features(data, questions, ("normal",) + TEN, (0,)), labels, fold
        ),
    }
    for name, value in probabilities.items():
        predictions[name] = decode(value)

    expected_three = probabilities["three"] @ np.arange(10)
    scores = {name: utility(value, labels) for name, value in predictions.items()}
    gate = paired(scores["ten"], scores["three"], players)
    report = {
        "cohort": {
            "games": int(len(data["labels"])),
            "players": int(len(np.unique(data["players"]))),
            "questions": int(len(labels)),
            "streams": list(data["streams"]),
        },
        "arms": {name: summary(value, labels) for name, value in predictions.items()},
        "paired_vs_three": {
            name: paired(scores[name], scores["three"], players)
            for name in predictions if name != "three"
        },
        "redundancy_spearman_vs_three_expected_rank": {
            "rankmle_argmax": round(float(spearmanr(predictions["rankmle"], expected_three)[0]), 4),
            "rankmle_centered_argmax": round(
                float(spearmanr(predictions["rankmle_centered"], expected_three)[0]), 4
            ),
            "ten_expected_rank": round(
                float(spearmanr(probabilities["ten"] @ np.arange(10), expected_three)[0]), 4
            ),
        },
        "rankmle_prediction_histogram": np.bincount(
            predictions["rankmle"], minlength=10
        ).tolist(),
        "gate": {
            "rule": "ten - three >= +0.003 and paired 95% lower bound > 0",
            **gate,
            "passes": bool(gate["delta"] >= 0.003 and gate["ci95"][0] > 0),
        },
    }
    text = json.dumps(report, indent=1)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
