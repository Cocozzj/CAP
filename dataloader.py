"""
DataLoader for CAP.

Two datasets:
    DatasetA   : real Dataset-A (PartNet-Mobility + 3-cam mp4 + standard 3DGS PLY)
    ToyDataset : random-tensor stub for smoke-tests (used by eval/ scripts)

Per-sample dict (same shape from both):
    {
        "frames":     Tensor [V, T, 3, H, W]  float32 in [0, 1]
        "gs_params":  GSParameter             (5 tensor fields, t=0)
        "text":       str                     verb phrase
        "intrinsics": Tensor [V, 3, 3]        (DatasetA only)
        "extrinsics": Tensor [V, 4, 4]        (DatasetA only, world→cam OpenCV)
        "task_id":    int                     (DatasetA only, sorted task index)
    }

Hard constraints: V≥1, T≥10 and T%5==0, frames in [0, 1],
mu in WORLD frame, mu/scale/opacity time-aligned to t=0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from model.utils import GSParameter
from dataset_a_text import task_to_text


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
# PLY loading — standard 3DGS only (x, y, z, [nx, ny, nz,] f_dc_*, opacity,
# scale_*, rot_*).  SH degrees missing in the PLY (e.g. only DC) are zero-filled.
# ============================================================================
def load_init_gs_ply(ply_path: Path,
                     n_points: Optional[int] = 10000,
                     seed: int = 0,
                     c_sh: int = 48) -> GSParameter:
    """Read init_gs.ply → GSParameter; subsample to ``n_points``.

    ``c_sh`` is the SH coefficient count expected by the model
    (= ``cfg.gs_param.gs_dimension - 11`` ; default 48 for SH degree 3 RGB).
    Higher orders missing in the PLY are filled with zeros.
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
# Camera loading
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
# DatasetA
# ============================================================================
class DatasetA(Dataset):
    """Read manifest.json + per-trajectory dirs (mp4 × 3 + init_gs.ply +
    cameras.json + meta.json) → per-sample dict.

    Args:
        manifest_path: outputs/manifest.json
        data_dir:      outputs/data
        split:         entry must contain ``split`` in its ``splits`` list
        T:             frames per sample (>=10, divisible by 5; default 30)
        image_size:    resize each frame to this square size (default 256)
        n_gs_points:   subsample init_gs.ply to this many (default 10000)
        c_sh:          SH coefficient count (default 48 = degree 3 RGB)
    """

    CAMERAS = ("cam0", "cam1", "cam2")

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
        e = self.entries[i]
        d = self.data_dir / e["rel_dir"]
        idx = np.linspace(0, e["n_frames"] - 1, self.T, dtype=int).tolist()

        frames = torch.stack([
            _load_video_frames(d / f"{cam}.mp4", idx, target_size=self.image_size)
            for cam in self.CAMERAS
        ], dim=0).clamp(0.0, 1.0)                                         # [V, T, 3, H, W]

        gs = load_init_gs_ply(d / "init_gs.ply",
                              n_points=self.n_gs_points,
                              seed=i, c_sh=self.c_sh)

        K, w2c = load_cameras(d / "cameras.json", self.CAMERAS)

        return {
            "frames":     frames,
            "gs_params":  gs,
            "text":       task_to_text(e["task_name"], e["obj_category"]),
            "intrinsics": K,
            "extrinsics": w2c,
            "task_id":    self.task_to_id[e["task_name"]],
        }


# ============================================================================
# ToyDataset — random-tensor stub used by eval/ smoke-tests
# ============================================================================
class ToyDataset(Dataset):
    """Tiny random-tensor dataset matching CAPModel.forward shapes."""

    def __init__(self, n_samples: int = 16, n_views: int = 3, n_frames: int = 30,
                 img_size: int = 64, n_gaussians: int = 256, sh_dim: int = 48):
        if n_frames < 10 or n_frames % 5 != 0:
            raise ValueError(f"n_frames must be >= 10 and divisible by 5; got {n_frames}")
        self.n_samples, self.V, self.T = n_samples, n_views, n_frames
        self.H = self.W = img_size
        self.Ng, self.sh_dim = n_gaussians, sh_dim

    def __len__(self): return self.n_samples

    def __getitem__(self, i: int) -> Dict:
        torch.manual_seed(i)
        return {
            "frames":    torch.rand(self.V, self.T, 3, self.H, self.W),
            "gs_params": GSParameter(
                mu      = torch.randn(self.Ng, 3) * 0.5,
                cov     = torch.randn(self.Ng, 4),
                scale   = torch.full((self.Ng, 3), -3.0) + torch.randn(self.Ng, 3) * 0.1,
                sh      = torch.randn(self.Ng, self.sh_dim) * 0.1,
                opacity = torch.sigmoid(torch.randn(self.Ng, 1)),
            ),
            "text": "open the drawer",
        }


# ============================================================================
# Collate
# ============================================================================
def collate_fn(batch: List[Dict]) -> Dict:
    """Stack fixed-shape tensors; keep per-sample GSParameter list (N varies)."""
    out: Dict = {
        "frames":    torch.stack([b["frames"] for b in batch], dim=0),
        "gs_params": [b["gs_params"] for b in batch],
        "text":      [b["text"] for b in batch],
    }
    for k in ("intrinsics", "extrinsics"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
    return out


# Backward-compat alias.
collate_batch = collate_fn


__all__ = [
    "DatasetA", "ToyDataset",
    "collate_fn", "collate_batch",
    "GSParameter", "task_to_text",
    "load_init_gs_ply", "load_cameras",
]


# ============================================================================
# Smoke-test (run with: python dataloader.py path/to/manifest.json path/to/data)
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        ds = DatasetA(sys.argv[1], sys.argv[2], split="train")
    else:
        ds = ToyDataset(n_samples=4)
    s = ds[0]
    print(f"Dataset len={len(ds)}  frames={tuple(s['frames'].shape)}  "
          f"gs.N={len(s['gs_params'])}  text={s['text']!r}")
