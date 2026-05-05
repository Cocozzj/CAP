"""DatasetB — Something Something v2 real video, 1-camera + MiDaS pseudo-depth.

Per-trajectory dir layout:
    data/dataset_b/data/<traj_id>/
        cam0.mp4          (single real-world video)
        cameras.json      (1 cam: synthetic pinhole pose, identity extrinsics)
        init_gs.ply       (3DGS extracted from first frame, e.g. mvsplat)
        depth.npz         (MiDaS depth on the FIRST frame only — static, [H, W])
        meta.json         (raw_label / template / placeholders + n_frames + ...)

Differences vs DatasetA:
  - V = 1 (real video has only one camera)
  - No GT physics or trajectory (real video → no simulator)
  - depth is a SINGLE static map (initial frame), not per-timestep
  - text is already natural English (raw_label / template)

The single static depth is exposed as a [V=1, H, W] tensor; loss-side code
should compare it only against ``rendered_depth`` at the initial timestep
(timestep_index=0 in the trajectory render).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .common import _load_video_frames, load_cameras, load_init_gs_ply
from .text import dataset_b_text


class DatasetB(Dataset):
    """Read manifest.json + per-trajectory dirs → per-sample dict.

    Args:
        manifest_path: path to ``data/dataset_b/manifest.json``
        data_dir:      path to ``data/dataset_b/data``
        split:         "train" / "val" / "test"
        T:             frames per sample (>=10, divisible by 5; default 30)
        image_size:    resize each frame to this square size (default 256)
        n_gs_points:   subsample init_gs.ply (default 10000)
        c_sh:          SH coefficient count (default 48 = degree 3 RGB)
        load_depth:    True → include MiDaS depth in the per-sample dict
                       (False saves 256² * 4 bytes per sample if you don't use it)
    """

    CAMERAS = ("cam0",)                         # V=1

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir:      Union[str, Path],
        split:         str = "train",
        T:             int = 30,
        image_size:    int = 256,
        n_gs_points:   int = 10000,
        c_sh:          int = 48,
        load_depth:    bool = True,
    ):
        if T < 10 or T % 5 != 0:
            raise ValueError(f"T must be >= 10 and a multiple of 5; got {T}")

        self.data_dir    = Path(data_dir)
        self.T           = T
        self.image_size  = image_size
        self.n_gs_points = n_gs_points
        self.c_sh        = c_sh
        self.split       = split
        self.load_depth  = load_depth

        with open(manifest_path) as f:
            entries = json.load(f)["entries"]
        # DatasetB entries also use ``splits`` list (same convention as A).
        self.entries = [e for e in entries if split in e.get("splits", [])]
        if not self.entries:
            avail = sorted({s for e in entries for s in e.get("splits", [])})
            raise ValueError(f"No entries for split={split!r}.  Available: {avail}")

        # Stable task_id mapping (per-split; same convention as DatasetA)
        unique_tasks = sorted({e["task_name"] for e in self.entries})
        self.task_to_id = {t: i for i, t in enumerate(unique_tasks)}

    def __len__(self) -> int:
        return len(self.entries)

    def _load_depth_static(self, depth_path: Path) -> Optional[torch.Tensor]:
        """Load MiDaS static depth → [V=1, H, W] float32, resized to image_size.
        Returns None if the file doesn't exist (some entries may be missing it).
        """
        if not depth_path.exists():
            return None
        z = np.load(depth_path)
        # MiDaS variant typically writes "depth" (HxW float32)
        key = "depth" if "depth" in z.files else z.files[0]
        arr = np.asarray(z[key], dtype=np.float32)              # expected [H, W]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None                                         # unexpected shape — skip

        if arr.shape[-1] != self.image_size:
            from PIL import Image
            arr = np.asarray(
                Image.fromarray(arr).resize((self.image_size, self.image_size),
                                             Image.BILINEAR),
                dtype=np.float32,
            )
        return torch.from_numpy(arr).unsqueeze(0)               # [V=1, H, W]

    def __getitem__(self, i: int) -> Dict:
        e   = self.entries[i]
        d   = self.data_dir / e["rel_dir"]
        idx = np.linspace(0, e["n_frames"] - 1, self.T, dtype=int).tolist()

        frames = torch.stack([
            _load_video_frames(d / f"{cam}.mp4", idx, target_size=self.image_size)
            for cam in self.CAMERAS
        ], dim=0).clamp(0.0, 1.0)                                   # [V=1, T, 3, H, W]

        gs = load_init_gs_ply(d / "init_gs.ply",
                              n_points=self.n_gs_points,
                              seed=i, c_sh=self.c_sh)

        K, w2c = load_cameras(d / "cameras.json", self.CAMERAS)

        out: Dict = {
            "frames":     frames,
            "gs_params":  gs,
            "text":       dataset_b_text(e),
            "intrinsics": K,                                        # [1, 3, 3]
            "extrinsics": w2c,                                      # [1, 4, 4]
            "task_id":    self.task_to_id[e["task_name"]],
        }

        if self.load_depth:
            depth = self._load_depth_static(d / "depth.npz")
            if depth is not None:
                out["depth"] = depth                                # [V=1, H, W]
        return out


__all__ = ["DatasetB"]
