"""Manifest writer for Dataset-B. Output schema mirrors Dataset-A so the
shared dataloader can read either dataset (or a concatenation) without
branching on object_type.

Each manifest entry has:
    traj_id, obj_id, obj_category, task_name, n_frames, image_size,
    split (canonical), splits (list — for Dataset-B always single-element),
    rel_dir, source ('ssv2'|'mpii'|...), object_type ('real_video').
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def write_manifest(
    data_dir: str | Path,
    splits: Dict[str, List[str]],
    out_path: str | Path,
) -> None:
    """Walk data_dir, read each meta.json, attach split membership, write
    a single manifest.json compatible with the shared dataloader."""
    data_dir = Path(data_dir)

    traj_to_split: Dict[str, str] = {}
    for split_name, ids in splits.items():
        for tid in ids:
            traj_to_split[tid] = split_name

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
        split = traj_to_split.get(tid, "unassigned")
        entries.append({
            "traj_id":      tid,
            "obj_id":       meta.get("obj_id", ""),
            "obj_category": meta.get("obj_category", "ssv2_realworld"),
            "task_name":    meta["task_name"],
            "n_frames":     meta["n_frames"],
            "image_size":   meta["image_size"],
            "split":        split,
            "splits":       [split] if split != "unassigned" else [],
            "rel_dir":      str(traj_dir.relative_to(data_dir)),
            "source":       meta.get("source", "ssv2"),
            "object_type":  meta.get("object_type", "real_video"),
            "raw_label":    meta.get("raw_label", ""),
        })

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"entries": entries, "n": len(entries)}, f, indent=2)
    logger.info("Wrote manifest with %d entries to %s", len(entries), out_path)
