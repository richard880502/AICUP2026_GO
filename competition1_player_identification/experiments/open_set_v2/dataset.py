"""Loads bounded move caches and builds content-disjoint v2 cohorts.

Example:
    python -m competition1_player_identification.experiments.open_set_v2.dataset

All row indices refer to shards concatenated in the documented RANKS order.
"""

import hashlib
import json
import pathlib

import numpy as np

from competition1_player_identification.experiments.open_set_v2 import artifacts
from competition1_player_identification.experiments.open_set_v2 import cache

ROOT = pathlib.Path("production/open_set_v2_20261111")


def load_arrays(root: pathlib.Path, keys: tuple[str, ...]) -> dict:
    """Concatenates complete rank shards in a fixed, rank-independent order."""
    parts: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for rank in cache.RANKS:
        with np.load(root / f"{rank}.npz") as shard:
            for key in keys:
                parts[key].append(shard[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def player_groups(players: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Groups arbitrary game order into sorted, contiguous player groups."""
    identifiers, inverse = np.unique(players, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return identifiers, list(np.split(order, boundaries))


def choose_cohort(
    groups: list[np.ndarray],
    hashes: np.ndarray,
    protected: set[str],
    generator: np.random.Generator,
    excluded: set[int],
) -> tuple[np.ndarray, set[int]]:
    """Samples 600 players with globally unique games, ten refs and five
    queries.
    """
    selected = []
    owners = set()
    for owner in generator.permutation(len(groups)):
        if int(owner) in excluded:
            continue
        records = generator.permutation(groups[owner])
        available = []
        seen: set[str] = set()
        for index in records:
            digest = str(hashes[index])
            if digest not in protected and digest not in seen:
                available.append(int(index))
                seen.add(digest)
            if len(available) == 15:
                break
        if len(available) == 15:
            selected.append(available)
            owners.add(int(owner))
            protected.update(str(hashes[index]) for index in available)
        if len(selected) == 600:
            return np.asarray(selected, dtype=np.int64), owners
    raise ValueError(
        f"Only {len(selected)} players have 15 content-unique games"
    )


def select_evaluation(arrays: dict, seed: int) -> tuple:
    """Selects content-disjoint evaluation players from mixed-color players."""
    identifiers, all_groups = player_groups(arrays["players"])
    eligible = []
    eligible_owners = []
    for owner, indices in enumerate(all_groups):
        if (
            len(indices) >= 15
            and np.bincount(
                arrays["colors"][indices],
                minlength=2,
            ).min()
            >= 5
        ):
            eligible.append(indices)
            eligible_owners.append(owner)
    generator = np.random.default_rng(seed)
    protected: set[str] = set()
    validation, validation_owners = choose_cohort(
        eligible,
        arrays["hashes"],
        protected,
        generator,
        set(),
    )
    final, final_owners = choose_cohort(
        eligible,
        arrays["hashes"],
        protected,
        generator,
        validation_owners,
    )
    # Exclude previously selected identities even if different games remain.
    if validation_owners & final_owners:
        raise ValueError(
            "Player overlap: cohort selection must exclude identities"
        )
    held_out_owners = {
        eligible_owners[owner] for owner in validation_owners | final_owners
    }
    return (
        validation,
        final,
        identifiers[sorted(held_out_owners)],
        protected,
        len(identifiers),
    )


def build_splits(root: pathlib.Path, seed: int) -> dict:
    """Excludes evaluation game content from the entire training population."""
    target = root / "clean_splits.npz"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite locked splits: {target}")
    arrays = load_arrays(
        root / "raw_cache",
        (
            "players",
            "hashes",
            "colors",
        ),
    )
    validation, final, held_out_players, protected, count = select_evaluation(
        arrays,
        seed,
    )
    train = np.flatnonzero(
        ~np.isin(arrays["players"], held_out_players)
        & ~np.isin(arrays["hashes"], list(protected)),
    )
    payload = {
        "train": train,
        "validation": validation,
        "final_holdout": final,
        "validation_players": arrays["players"][validation[:, 0]],
        "final_holdout_players": arrays["players"][final[:, 0]],
    }
    with target.open("wb") as output:
        np.savez_compressed(
            output,
            train=payload["train"],
            validation=payload["validation"],
            final_holdout=payload["final_holdout"],
            validation_players=payload["validation_players"],
            final_holdout_players=payload["final_holdout_players"],
        )
    report = make_report(
        arrays,
        payload,
        count,
        seed,
    )
    (root / "split_audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def make_report(
    arrays: dict,
    splits: dict,
    player_count: int,
    seed: int,
) -> dict:
    """Measures actual content separation, colors, and immutable split hashes."""
    hash_sets = {
        name: set(arrays["hashes"][indices].ravel().tolist())
        for name, indices in splits.items()
        if name
        in (
            "train",
            "validation",
            "final_holdout",
        )
    }
    overlap = {
        "train_validation": len(hash_sets["train"] & hash_sets["validation"]),
        "train_final": len(hash_sets["train"] & hash_sets["final_holdout"]),
        "validation_final": len(
            hash_sets["validation"] & hash_sets["final_holdout"]
        ),
    }
    if any(overlap.values()):
        raise ValueError(f"Game-content leakage: {overlap!r}")
    player_sets = {
        name: set(arrays["players"][splits[name]].ravel().tolist())
        for name in hash_sets
    }
    player_overlap = {
        "train_validation": len(
            player_sets["train"] & player_sets["validation"]
        ),
        "train_final": len(player_sets["train"] & player_sets["final_holdout"]),
        "validation_final": len(
            player_sets["validation"] & player_sets["final_holdout"]
        ),
    }
    if any(player_overlap.values()):
        raise ValueError(f"Player leakage: {player_overlap!r}")
    for name in ("validation", "final_holdout"):
        if splits[name].shape != (600, 15) or len(hash_sets[name]) != 9000:
            raise ValueError(f"Invalid or duplicated evaluation games: {name}")
    return {
        "seed": seed,
        "total_games": len(arrays["players"]),
        "total_players": player_count,
        "training_games": len(splits["train"]),
        "training_players": len(np.unique(arrays["players"][splits["train"]])),
        "validation_players": 600,
        "final_holdout_players": 600,
        "canonical_game_hash_overlaps": overlap,
        "player_overlaps": player_overlap,
        "white_fraction": {
            name: float(arrays["colors"][indices].mean())
            for name, indices in splits.items()
            if name in hash_sets
        },
        "indices_sha256": {
            name: hashlib.sha256(indices.tobytes()).hexdigest()
            for name, indices in splits.items()
        },
        "status": "passed",
    }


def verify_locked_splits(root: pathlib.Path) -> dict:
    """Rechecks immutable split contents and hashes without resampling them."""
    expected = json.loads((root / "split_audit.json").read_text())
    arrays = load_arrays(root / "raw_cache", ("players", "hashes", "colors"))
    with np.load(root / "clean_splits.npz") as payload:
        splits = {key: np.asarray(payload[key]) for key in payload.files}
    report = make_report(
        arrays,
        splits,
        len(np.unique(arrays["players"])),
        expected["seed"],
    )
    if report["indices_sha256"] != expected["indices_sha256"]:
        raise ValueError("The locked split indices have changed")
    (root / "split_audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Builds clean splits after all ten immutable cache shards exist."""
    parser = artifacts.root_parser(__doc__, ROOT)
    artifacts.add_seed(parser)
    arguments = parser.parse_args()
    print(json.dumps(build_splits(arguments.root, arguments.seed), indent=2))


if __name__ == "__main__":
    main()
