"""DatasetA — PartNet-Mobility synthetic, 3-camera mp4 + standard 3DGS PLY.

Per-trajectory dir layout:
    data/dataset_a/data/<traj_id>/
        cam0.mp4, cam1.mp4, cam2.mp4    (3 viewpoint videos)
        cameras.json                     (3 cams: intrinsics + world→cam)
        init_gs.ply                      (standard 3DGS PLY at t=0)
        meta.json                        (task_name, obj_category, n_frames, ...)
        physics.json, trajectory.npz     (NOT loaded — PDF method is self-supervised)

PDF method uses only frames + text labels; physics.json / trajectory.npz are
SAPIEN simulator side-products and are deliberately ignored to keep the method
transferable to DatasetB (real video without sim GT).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import _load_video_frames, load_cameras, load_init_gs_ply
from .text import task_to_text


class DatasetA(Dataset):
    """Read manifest.json + per-trajectory dirs → per-sample dict.

    Args:
        manifest_path: path to ``data/dataset_a/manifest.json``
        data_dir:      path to ``data/dataset_a/data``
        split:         entry must contain ``split`` in its ``splits`` list
                       (e.g. "train", "val", "test_iid", "test_ood_unseen_pair", ...)
        T:             frames per sample (>=10, divisible by 5; default 30)
        image_size:    resize each frame to this square size (default 256)
        n_gs_points:   subsample init_gs.ply to this many (default 10000)
        c_sh:          SH coefficient count (default 48 = degree 3 RGB)
    """

    CAMERAS = ("cam0", "cam1", "cam2")          # V=3

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir:      Union[str, Path],
        split:         str = "train",
        T:             int = 30,
        image_size:    int = 256,
        n_gs_points:   int = 10000,
        c_sh:          int = 48,
    ):
        if T < 10 or T % 5 != 0:
            raise ValueError(f"T must be >= 10 and a multiple of 5; got {T}")

        self.data_dir    = Path(data_dir)
        self.T           = T
        self.image_size  = image_size
        self.n_gs_points = n_gs_points
        self.c_sh        = c_sh
        self.split       = split

        with open(manifest_path) as f:
            entries = json.load(f)["entries"]
        self.entries = [e for e in entries if split in e.get("splits", [])]
        if not self.entries:
            avail = sorted({s for e in entries for s in e.get("splits", [])})
            raise ValueError(f"No entries for split={split!r}.  Available: {avail}")

        # Stable task_id mapping (sorted task_name index)
        unique_tasks = sorted({e["task_name"] for e in self.entries})
        self.task_to_id = {t: i for i, t in enumerate(unique_tasks)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        e   = self.entries[i]
        d   = self.data_dir / e["rel_dir"]
        idx = np.linspace(0, e["n_frames"] - 1, self.T, dtype=int).tolist()

        frames = torch.stack([
            _load_video_frames(d / f"{cam}.mp4", idx, target_size=self.image_size)
            for cam in self.CAMERAS
        ], dim=0).clamp(0.0, 1.0)                                   # [V=3, T, 3, H, W]

        gs = load_init_gs_ply(d / "init_gs.ply",
                              n_points=self.n_gs_points,
                              seed=i, c_sh=self.c_sh)

        K, w2c = load_cameras(d / "cameras.json", self.CAMERAS)

        return {
            "frames":     frames,
            "gs_params":  gs,
            "text":       task_to_text(e["task_name"], e["obj_category"]),
            "intrinsics": K,                                        # [3, 3, 3]
            "extrinsics": w2c,                                      # [3, 4, 4]
            "task_id":    self.task_to_id[e["task_name"]],
        }


__all__ = ["DatasetA"]
