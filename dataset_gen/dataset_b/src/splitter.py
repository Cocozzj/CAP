"""Train/val/test split for Dataset-B.

Simpler than Dataset-A's splitter — Dataset-B is a single-source dataset
without (object × task) pair structure or held-out categories. We do a
straightforward stratified random split per verb so every split sees all
8 verbs, and use the seed for reproducibility.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def split_by_verb(
    records: List[dict],
    *,
    train_frac: float = 0.80,
    val_frac:   float = 0.10,
    test_frac:  float = 0.10,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Stratified train/val/test split per verb.

    Each `record` must have keys: traj_id, task_name (= our_verb).
    Returns dict {split_name: [traj_id, ...]}.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, (
        f"fractions must sum to 1; got {train_frac}+{val_frac}+{test_frac}"
    )
    rng = random.Random(seed)

    by_verb: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        by_verb[r["task_name"]].append(r["traj_id"])

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for verb in sorted(by_verb):
        ids = by_verb[verb]
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * train_frac))
        n_va = int(round(n * val_frac))
        n_te = n - n_tr - n_va
        splits["train"].extend(ids[:n_tr])
        splits["val"].extend(ids[n_tr: n_tr + n_va])
        splits["test"].extend(ids[n_tr + n_va:])
        logger.info("verb=%s : train=%d val=%d test=%d", verb, n_tr, n_va, n_te)

    return splits


def save_splits(splits: Dict[str, List[str]], out_path: str | Path,
                meta: dict | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "splits": splits,
        "n_per_split": {k: len(v) for k, v in splits.items()},
    }
    if meta:
        payload["meta"] = meta
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote splits to %s", out_path)


def assert_no_leakage(splits: Dict[str, List[str]]) -> None:
    """A traj_id should appear in exactly one split."""
    seen = set()
    dup = []
    for name, ids in splits.items():
        for tid in ids:
            if tid in seen:
                dup.append((name, tid))
            seen.add(tid)
    assert not dup, f"Leakage: {dup[:5]}"
    logger.info("No-leakage check passed (%d unique trajectory ids)", len(seen))
