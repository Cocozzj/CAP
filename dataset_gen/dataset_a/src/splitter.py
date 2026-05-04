"""Train/val/test splits with held-out (object_class × task) pairs for OOD
compositional generalization.

Also constructs the Dataset-D split (held-out 3 categories entirely).
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def split_trajectories(
    records: List[dict],
    *,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_iid_frac: float = 0.10,
    ood_pair_fraction: float = 0.10,
    held_out_categories: List[str] | None = None,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Returns dict of split_name -> list of traj_ids.

    Splits produced:
        train, val, test_iid          (in-distribution)
        test_ood_unseen_pair          (held-out object_class × task)
        test_ood_unseen_object        (held-out object instance, seen pair)
        dataset_d_train, dataset_d_test  (held-out categories entirely)
    """
    held_out_categories = held_out_categories or []
    rng = random.Random(seed)

    # ---- 1. enumerate (category × task) pairs
    pair_to_ids: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for r in records:
        if not r.get("success", True):
            # we keep failed traj ids out of all splits
            continue
        key = (r["obj_category"], r["task_name"])
        pair_to_ids[key].append(r["traj_id"])

    # exclude pairs whose category is being held out wholesale (Dataset-D)
    eligible_pairs = [p for p in pair_to_ids
                      if p[0] not in held_out_categories and len(pair_to_ids[p]) >= 2]

    # ---- 2. choose held-out pairs from eligible (multi-instance, in-category) pairs
    n_held = max(1, int(len(eligible_pairs) * ood_pair_fraction))
    rng.shuffle(eligible_pairs)
    held_out_pairs: Set[Tuple[str, str]] = set(eligible_pairs[:n_held])

    # ---- 3. choose held-out object instances (within seen pairs)
    obj_to_ids: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        if not r.get("success", True):
            continue
        if (r["obj_category"], r["task_name"]) in held_out_pairs:
            continue
        obj_to_ids[r["obj_id"]].append(r["traj_id"])
    held_out_object_fraction = 0.05
    n_held_obj = max(1, int(len(obj_to_ids) * held_out_object_fraction))
    held_out_objs: Set[str] = set(rng.sample(list(obj_to_ids.keys()), n_held_obj))

    # ---- 4. assign each traj to a split
    splits: Dict[str, List[str]] = {
        "train": [],
        "val": [],
        "test_iid": [],
        "test_ood_unseen_pair": [],
        "test_ood_unseen_object": [],
        "test_compositional_long": [],     # NEW: eval-only multi-step compositions
        "dataset_d_train": [],
        "dataset_d_test": [],
    }

    for r in records:
        if not r.get("success", True):
            continue
        traj_id = r["traj_id"]
        cat = r["obj_category"]
        task = r["task_name"]
        obj_id = r["obj_id"]

        # eval-only compositions go to a dedicated split, regardless of category
        if r.get("eval_only", False):
            splits["test_compositional_long"].append(traj_id)
            continue

        # Dataset-D: separate from the main splits
        if cat in held_out_categories:
            splits["dataset_d_test"].append(traj_id)
            continue
        else:
            splits["dataset_d_train"].append(traj_id)

        if (cat, task) in held_out_pairs:
            splits["test_ood_unseen_pair"].append(traj_id)
            continue
        if obj_id in held_out_objs:
            splits["test_ood_unseen_object"].append(traj_id)
            continue

        # in-distribution split
        r_split = rng.random()
        cum_train = train_frac
        cum_val = train_frac + val_frac
        if r_split < cum_train:
            splits["train"].append(traj_id)
        elif r_split < cum_val:
            splits["val"].append(traj_id)
        else:
            splits["test_iid"].append(traj_id)

    _log_split_stats(splits)
    return splits


def _log_split_stats(splits: Dict[str, List[str]]):
    for name, ids in splits.items():
        logger.info("  %-30s %d", name, len(ids))


def save_splits(splits: Dict[str, List[str]], out_path: str | Path,
                meta: dict | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": splits, "n_per_split": {k: len(v) for k, v in splits.items()}}
    if meta:
        payload["meta"] = meta
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote splits to %s", out_path)


def assert_no_leakage(splits: Dict[str, List[str]],
                      records: List[dict]) -> None:
    """Sanity check: held-out objects/pairs must not appear in train/val/test_iid."""
    rec_by_id = {r["traj_id"]: r for r in records}

    def cats(name): return {rec_by_id[i]["obj_category"] for i in splits[name]}
    def objs(name): return {rec_by_id[i]["obj_id"] for i in splits[name]}
    def pairs(name): return {(rec_by_id[i]["obj_category"], rec_by_id[i]["task_name"])
                              for i in splits[name]}

    train_pairs = pairs("train")
    train_objs = objs("train")
    train_cats = cats("train")

    ood_pairs = pairs("test_ood_unseen_pair")
    ood_objs = objs("test_ood_unseen_object")
    d_test_cats = cats("dataset_d_test")

    assert train_pairs.isdisjoint(ood_pairs), \
        f"Pair leakage: {train_pairs & ood_pairs}"
    assert train_objs.isdisjoint(ood_objs), \
        f"Object leakage: {train_objs & ood_objs}"
    assert train_cats.isdisjoint(d_test_cats), \
        f"Category leakage: {train_cats & d_test_cats}"
    logger.info("No-leakage check passed.")
