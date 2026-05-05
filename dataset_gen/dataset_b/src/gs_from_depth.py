"""Build a 3DGS PLY init from a single RGB frame + estimated depth.

Output PLY uses the SAME field layout as Dataset-A's mesh-backend init_gs.ply
so the shared dataloader works without branching:
    x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity,
    scale_0, scale_1, scale_2, rot_0, rot_1, rot_2, rot_3
where:
    f_dc_*      = (rgb - 0.5) / 0.28209  (standard SH DC convention)
    opacity     = logit(0.95) = 2.9444   (matches Dataset-A)
    scale_*     = log(0.025) ~ -3.69     (slightly larger than mesh-backend's
                                          log(0.037)=-3.30 — depth-derived
                                          point clouds are sparser)
    rot_*       = (1, 0, 0, 0)           identity quaternion (wxyz)
    nx/ny/nz    = 0
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_SH_C0 = 0.28209479177387814
_OPACITY_LOGIT = 2.9444   # logit(0.95)
_SCALE_LOG = float(np.log(0.025))   # ~-3.69


def back_project(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    *,
    depth_min: float = 0.1,
    depth_max: float = 10.0,
    background_quantile: float = 0.95,
    n_points: int = 10000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Lift each pixel into 3D world (= camera) frame.

    Args:
        rgb:   (H, W, 3) uint8 OR float in [0,1]
        depth: (H, W) float32, meters
        K:     (3, 3) intrinsics
        depth_min / depth_max: drop pixels outside this depth range
        background_quantile: drop pixels with depth above this quantile
            (treats far pixels as 'sky/background')
        n_points: subsample to this many points
    Returns:
        xyz:    (n_points, 3) float32, points in camera/world frame (meters)
        rgb01:  (n_points, 3) float32, in [0, 1]
    """
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Per-pixel valid mask
    bg_thresh = float(np.quantile(depth, background_quantile))
    mask = (
        (depth >= depth_min) &
        (depth <= depth_max) &
        (depth <= bg_thresh) &
        np.isfinite(depth)
    )
    n_valid = int(mask.sum())
    if n_valid < 100:
        # Fallback: skip the BG quantile filter
        mask = (depth >= depth_min) & (depth <= depth_max) & np.isfinite(depth)
        n_valid = int(mask.sum())
    if n_valid == 0:
        raise RuntimeError("No valid depth pixels — depth map might be all NaN")

    ys, xs = np.where(mask)
    z = depth[ys, xs]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    # Camera frame: +Z forward (away from camera), +X right, +Y down.
    # Our PartNet/Dataset-A convention is also a right-handed frame, so we
    # leave coordinates as-is. The dataloader treats `mu` as world coords;
    # for single-view real video we define world = camera frame.
    xyz = np.stack([x, y, z], axis=-1).astype(np.float32)

    # rgb to [0, 1] floats
    if rgb.dtype == np.uint8:
        rgb01 = rgb.astype(np.float32) / 255.0
    else:
        rgb01 = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    rgb_at = rgb01[ys, xs]   # (n_valid, 3)

    # subsample
    if xyz.shape[0] > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(xyz.shape[0], size=n_points, replace=False)
        idx.sort()
        xyz = xyz[idx]
        rgb_at = rgb_at[idx]

    return xyz, rgb_at


def write_init_gs_ply(
    out_path: Path,
    xyz: np.ndarray,
    rgb01: np.ndarray,
) -> None:
    """Write the (xyz, rgb) point cloud as a Dataset-A-compatible 3DGS PLY."""
    try:
        from plyfile import PlyData, PlyElement
    except ImportError as e:
        raise RuntimeError("plyfile required. pip install plyfile") from e

    n = xyz.shape[0]
    f_dc = (rgb01 - 0.5) / _SH_C0   # SH DC for each channel

    structured = np.empty(n, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ])

    structured["x"]  = xyz[:, 0]
    structured["y"]  = xyz[:, 1]
    structured["z"]  = xyz[:, 2]
    structured["nx"] = 0.0
    structured["ny"] = 0.0
    structured["nz"] = 0.0
    structured["f_dc_0"] = f_dc[:, 0]
    structured["f_dc_1"] = f_dc[:, 1]
    structured["f_dc_2"] = f_dc[:, 2]
    structured["opacity"] = _OPACITY_LOGIT
    structured["scale_0"] = _SCALE_LOG
    structured["scale_1"] = _SCALE_LOG
    structured["scale_2"] = _SCALE_LOG
    structured["rot_0"] = 1.0
    structured["rot_1"] = 0.0
    structured["rot_2"] = 0.0
    structured["rot_3"] = 0.0

    el = PlyElement.describe(structured, "vertex")
    PlyData([el]).write(str(out_path))
