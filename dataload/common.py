"""Shared utilities used by both DatasetA and DatasetB.

  _load_video_frames : decord (preferred) / imageio mp4 reader
  load_init_gs_ply    : standard 3DGS .ply → GSParameter
  load_cameras        : cameras.json → (intrinsics, extrinsics) for any V
  collate_fn          : batch dict assembler (handles optional intrinsics /
                        extrinsics / task_id / depth fields per dataset)

Per-sample dict (output of __getitem__ in either dataset):
    {
        "frames":     Tensor [V, T, 3, H, W]  float32 in [0, 1]
        "gs_params":  GSParameter             (5 tensor fields, t=0)
        "text":       str                     natural-language action phrase
        "intrinsics": Tensor [V, 3, 3]
        "extrinsics": Tensor [V, 4, 4]        world→cam (OpenCV)
        "task_id":    int                     stable per-split task index
        "depth":      Tensor [V, H, W]        (DatasetB only — single MiDaS map)
    }

Hard constraints: V≥1, T≥10 and T%5==0, frames in [0, 1],
mu in WORLD frame, mu/scale/opacity time-aligned to t=0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from model.utils import GSParameter


# ============================================================================
# Video loading (decord preferred, imageio fallback)
# ============================================================================

def _load_video_frames(mp4_path: Path,
                       frame_indices: Sequence[int],
                       target_size: Optional[int] = None) -> torch.Tensor:
    """Load specific frames from an mp4 → [T, 3, H, W] float32 in [0, 1]."""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(mp4_path))
        arr = vr.get_batch(list(frame_indices)).asnumpy()       # (T, H, W, 3) uint8
    except ImportError:
        import imageio
        reader = imageio.get_reader(str(mp4_path))
        try:
            arr = np.stack([reader.get_data(int(i)) for i in frame_indices], axis=0)
        finally:
            reader.close()

    if target_size is not None and arr.shape[1] != target_size:
        from PIL import Image
        out = np.empty((arr.shape[0], target_size, target_size, 3), dtype=np.uint8)
        for t in range(arr.shape[0]):
            out[t] = np.array(
                Image.fromarray(arr[t]).resize((target_size, target_size),
                                                Image.BILINEAR)
            )
        arr = out

    arr = arr.astype(np.float32) / 255.0                         # (T, H, W, 3)
    arr = np.transpose(arr, (0, 3, 1, 2))                         # (T, 3, H, W)
    return torch.from_numpy(arr)


# ============================================================================
# PLY loading — standard Inria 3DGS layout
#   x, y, z, [nx, ny, nz,] f_dc_*, opacity, scale_*, rot_*
# SH degrees missing in the PLY (e.g. only DC) are zero-filled.
# ============================================================================

def load_init_gs_ply(ply_path: Path,
                     n_points: Optional[int] = 10000,
                     seed: int = 0,
                     c_sh: int = 48) -> GSParameter:
    """Read init_gs.ply → GSParameter; subsample to ``n_points``.

    ``c_sh`` is the SH coefficient count expected by the model
    (= ``cfg.gs_param.gs_dimension - 11`` ; default 48 for SH degree 3 RGB).
    """
    from plyfile import PlyData

    v = PlyData.read(str(ply_path))["vertex"].data
    n = len(v)

    mu      = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    scale   = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                       axis=-1).astype(np.float32)                # log-scale
    cov     = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                       axis=-1).astype(np.float32)                # quat (w, x, y, z)
    cov     = cov / (np.linalg.norm(cov, axis=-1, keepdims=True) + 1e-8)

    op      = np.asarray(v["opacity"], dtype=np.float32)
    if op.min() < -0.05 or op.max() > 1.05:                      # logit → sigmoid
        op = 1.0 / (1.0 + np.exp(-op))
    opacity = np.clip(op, 0.0, 1.0)[:, None]

    sh = np.zeros((n, c_sh), dtype=np.float32)
    for i in range(3):
        sh[:, i] = np.asarray(v[f"f_dc_{i}"], dtype=np.float32)
    n_rest = c_sh - 3
    for i in range(n_rest):
        key = f"f_rest_{i}"
        if key in v.dtype.names:
            sh[:, 3 + i] = np.asarray(v[key], dtype=np.float32)

    if n_points is not None and n > n_points:
        idx = np.random.default_rng(seed).choice(n, size=n_points, replace=False)
        idx.sort()
        mu, cov, scale, sh, opacity = mu[idx], cov[idx], scale[idx], sh[idx], opacity[idx]

    return GSParameter(
        mu      = torch.from_numpy(np.ascontiguousarray(mu)),
        cov     = torch.from_numpy(np.ascontiguousarray(cov)),
        scale   = torch.from_numpy(np.ascontiguousarray(scale)),
        sh      = torch.from_numpy(np.ascontiguousarray(sh)),
        opacity = torch.from_numpy(np.ascontiguousarray(opacity)),
    )


# ============================================================================
# Camera loading — generic over V (works for V=1 DatasetB and V=3 DatasetA)
# ============================================================================

def load_cameras(cameras_json_path: Path,
                 camera_names: Sequence[str]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (intrinsics [V, 3, 3], extrinsics [V, 4, 4] world→cam, OpenCV)."""
    with open(cameras_json_path) as f:
        cams = json.load(f)
    Ks, w2cs = [], []
    for name in camera_names:
        intr = cams[name]["intrinsics"]
        Ks.append(np.array([
            [intr["fx"], 0,           intr["cx"]],
            [0,           intr["fy"], intr["cy"]],
            [0,           0,           1],
        ], dtype=np.float32))
        w2cs.append(np.array(
            cams[name]["extrinsics"]["world_to_camera_4x4"], dtype=np.float32))
    return (torch.from_numpy(np.stack(Ks)),
            torch.from_numpy(np.stack(w2cs)))


# ============================================================================
# Collate — robust to optional fields (DatasetA has no depth, DatasetB has)
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict:
    """Stack fixed-shape tensors; keep per-sample GSParameter list (N varies)."""
    out: Dict = {
        "frames":    torch.stack([b["frames"] for b in batch], dim=0),
        "gs_params": [b["gs_params"] for b in batch],
        "text":      [b["text"] for b in batch],
    }
    for k in ("intrinsics", "extrinsics", "depth"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
    return out


# Backward-compat alias.
collate_batch = collate_fn


__all__ = [
    "_load_video_frames",
    "load_init_gs_ply",
    "load_cameras",
    "collate_fn", "collate_batch",
]
