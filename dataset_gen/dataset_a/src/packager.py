"""Package the rendered data into either a folder layout or WebDataset shards.

Folder mode (default): per-trajectory subdirectories already produced by the
renderer; this module just writes the splits and a manifest.

WebDataset mode: tar each split into shards for fast streaming.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Priority ordering for the manifest's `split` field. A trajectory may
# belong to multiple splits (e.g. dataset_d_train AND train); when emitting
# a single canonical split per entry, we pick the most specific one. The
# full list is also kept under `splits`.
_PRIMARY_SPLIT_PRIORITY = (
    "test_compositional_long",   # eval-only multi-step compositions
    "test_ood_unseen_pair",      # held-out (cat × task) pair
    "test_ood_unseen_object",    # held-out object instance
    "dataset_d_test",            # held-out category (Dataset-D test)
    "test_iid",
    "val",
    "train",
    "dataset_d_train",           # superset; falls back here
)


def _pick_primary_split(member_splits: List[str]) -> str:
    for s in _PRIMARY_SPLIT_PRIORITY:
        if s in member_splits:
            return s
    return member_splits[0] if member_splits else "unassigned"


def write_manifest(
    data_dir: str | Path,
    splits: Dict[str, List[str]],
    out_path: str | Path,
) -> None:
    """A manifest is a single JSON listing every trajectory and the split(s)
    it belongs to. Each entry has:
        split:  primary canonical split (one of the names above, ordered by
                priority — most specific test split wins over train/dataset_d_train)
        splits: full list of split names this trajectory belongs to (order-stable).

    Used by the training data loader; common patterns:
        train_entries = [e for e in m if e["split"] == "train"]
        dataset_d_train_entries = [e for e in m if "dataset_d_train" in e["splits"]]
    """
    from collections import defaultdict

    data_dir = Path(data_dir)

    # Multi-label: a traj can be in multiple splits (e.g. train + dataset_d_train).
    traj_to_splits: Dict[str, List[str]] = defaultdict(list)
    for split_name, ids in splits.items():
        for tid in ids:
            traj_to_splits[tid].append(split_name)

    entries = []
    for traj_dir in sorted(data_dir.iterdir()):
        if not traj_dir.is_dir():
            continue
        meta_path = traj_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        tid = meta["traj_id"]
        member_splits = traj_to_splits.get(tid, [])
        entries.append({
            "traj_id": tid,
            "obj_id": meta["obj_id"],
            "obj_category": meta["obj_category"],
            "task_name": meta["task_name"],
            "n_frames": meta["n_frames"],
            "image_size": meta["image_size"],
            "split": _pick_primary_split(member_splits),
            "splits": member_splits,
            "rel_dir": str(traj_dir.relative_to(data_dir)),
        })

    out_path = Path(out_path)
    with open(out_path, "w") as f:
        json.dump({"entries": entries, "n": len(entries)}, f, indent=2)
    logger.info("Wrote manifest with %d entries to %s", len(entries), out_path)


def pack_webdataset(
    data_dir: str | Path,
    splits: Dict[str, List[str]],
    out_dir: str | Path,
    shards_per_split: int = 40,
) -> None:
    """Tar trajectories into shards, one set of shards per split."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in splits.items():
        if not ids:
            continue
        chunk = max(1, len(ids) // shards_per_split + 1)
        shard_idx = 0
        cur = []
        for tid in ids:
            cur.append(tid)
            if len(cur) >= chunk:
                _write_shard(data_dir, cur, out_dir, split_name, shard_idx)
                cur = []
                shard_idx += 1
        if cur:
            _write_shard(data_dir, cur, out_dir, split_name, shard_idx)


def _write_shard(data_dir: Path, ids: List[str], out_dir: Path,
                 split_name: str, idx: int) -> None:
    shard_path = out_dir / f"{split_name}-shard-{idx:05d}.tar"
    with tarfile.open(shard_path, "w") as tf:
        for tid in ids:
            traj_dir = data_dir / tid
            if not traj_dir.exists():
                continue
            for f in traj_dir.iterdir():
                arcname = f"{tid}/{f.name}"
                tf.add(f, arcname=arcname)
    logger.info("Wrote shard %s (%d trajectories)", shard_path, len(ids))
