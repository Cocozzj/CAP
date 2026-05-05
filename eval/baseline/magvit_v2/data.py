"""Per-trajectory video dataset for MAGVIT v2 training/inference.

Loads cam0.mp4 (or cam1/cam2 independently) at downsampled resolution.
Returns [T, 3, H, W] tensor.  Text instruction is also returned for the
text-conditioned transformer stage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from dataload.common import _load_video_frames


class MAGVITVideoDataset(Dataset):
    """One sample = one (trajectory, camera) pair → [T, 3, H, W] video.

    For Dataset-A (V=3), set ``cam_index=0`` to use cam0 (cam1/cam2 use
    independent training to match MAGVIT's per-camera setup).
    """
    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir:      Union[str, Path],
        split:         str = "train",
        T:             int = 20,          # ↓ from 30 (simplified for speed)
        image_size:    int = 64,          # ↓ from 128 (4× faster per iter)
        cam_index:     int = 0,           # which camera (0/1/2 for A, only 0 for B)
    ):
        self.data_dir   = Path(data_dir)
        self.T          = T
        self.image_size = image_size
        self.cam_index  = cam_index
        self.split      = split

        with open(manifest_path) as f:
            entries = json.load(f)["entries"]
        self.entries = [e for e in entries if split in e.get("splits", [])]
        if not self.entries:
            avail = sorted({s for e in entries for s in e.get("splits", [])})
            raise ValueError(f"No entries for split={split!r}.  Available: {avail}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> dict:
        from dataload.text import task_to_text

        e = self.entries[i]
        d = self.data_dir / e["rel_dir"]
        n_frames = int(e.get("n_frames", 30))
        idx = np.linspace(0, n_frames - 1, self.T, dtype=int).tolist()

        video_path = d / f"cam{self.cam_index}.mp4"
        frames = _load_video_frames(
            video_path, idx, target_size=self.image_size,
        ).clamp(0.0, 1.0)                                          # [T, 3, H, W]

        text = task_to_text(e["task_name"], e.get("obj_category", ""))
        return {
            "video":   frames,
            "text":    text,
            "traj_id": Path(e["rel_dir"]).name,
        }


def collate_magvit(batch: Sequence[dict]) -> dict:
    return {
        "video":   torch.stack([b["video"] for b in batch], dim=0),    # [B, T, 3, H, W]
        "texts":   [b["text"] for b in batch],
        "traj_id": [b["traj_id"] for b in batch],
    }
