from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SPLIT_NAMES = [
    "train",
    "val",
    "test_in_dist",
    "test_unseen_obj",
    "test_unseen_comb",
    "test_phys_ood",
]


def empty_splits() -> dict[str, list[str]]:
    return {name: [] for name in SPLIT_NAMES}


def deterministic_sample(items: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    count = int(round(len(shuffled) * fraction))
    selected = sorted(shuffled[:count])
    remaining = sorted(shuffled[count:])
    return selected, remaining


def write_splits(path: Path, splits: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2)
        handle.write("\n")
