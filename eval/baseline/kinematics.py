"""Shared kinematic utilities used by multiple baselines.

These were originally in eval.baseline.tamp_rule but are not TAMP-specific —
they're general-purpose 3DGS pose-application helpers.  Pulled out here so
that the (deprecated) tamp_rule package can be safely deleted.

Functions:
  quat_log_scale_to_full_cov   convert (quat[4], log_scale[3]) → full 3x3 cov
  quat_xyzw_to_R               quaternion → rotation matrix
  apply_pose_trajectory_to_gs  apply per-timestep SE(3) pose to Gaussians
                                returning [T, N, 3] / [T, N, 3, 3] etc.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Quaternion / cov utilities
# ──────────────────────────────────────────────────────────────────────

def quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion → 3×3 rotation matrix."""
    x, y, z, w = q[0], q[1], q[2], q[3]
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def quat_log_scale_to_full_cov(cov_quat: np.ndarray, log_scale: np.ndarray) -> np.ndarray:
    """Convert (quaternion[4], log_scale[3]) → full 3×3 covariance.

    init_gs.ply stores (quat=[w, x, y, z], log_scale).  Standard 3DGS formula:
        Σ = R diag(s)^2 R^T,    where R is from quaternion, s = exp(log_scale)
    """
    w, x, y, z = cov_quat[..., 0], cov_quat[..., 1], cov_quat[..., 2], cov_quat[..., 3]
    R = np.stack([
        np.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)], axis=-1),
        np.stack([    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)], axis=-1),
        np.stack([    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)], axis=-1),
    ], axis=-2).astype(np.float32)
    s = np.exp(log_scale).astype(np.float32)
    S = np.zeros(s.shape + (3,), dtype=np.float32)
    S[..., 0, 0] = s[..., 0]
    S[..., 1, 1] = s[..., 1]
    S[..., 2, 2] = s[..., 2]
    return R @ S @ S @ np.swapaxes(R, -1, -2)


# ──────────────────────────────────────────────────────────────────────
# SE(3) → GS application
# ──────────────────────────────────────────────────────────────────────

def apply_pose_trajectory_to_gs(
    mu0:        np.ndarray,            # [N, 3]
    cov0:       np.ndarray,            # [N, 3, 3]
    sh0:        np.ndarray,            # [N, C_sh]
    opacity0:   np.ndarray,            # [N, 1]
    scale0:     np.ndarray,            # [N, 3]
    poses:      np.ndarray,            # [T, 7]   xyz + xyzw quaternion
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply a per-timestep SE(3) pose to all Gaussians.

    The pose at t=0 (poses[0]) is treated as the reference; each subsequent
    pose's relative transform Δ = pose_t ∘ pose_0⁻¹ is applied to all Gaussians.

    Returns (mu_t, cov_t, sh_t, opacity_t, scale_t) shaped:
      mu_t      [T, N, 3]
      cov_t     [T, N, 3, 3]
      sh_t      [T, N, C_sh]      broadcast (no SH rotation)
      opacity_t [T, N, 1]         broadcast
      scale_t   [T, N, 3]         broadcast
    """
    T = int(poses.shape[0])
    N = int(mu0.shape[0])

    t0_off = poses[0, :3]                                                    # [3]
    R0     = quat_xyzw_to_R(poses[0, 3:])                                    # [3, 3]

    mu_t  = np.zeros((T, N, 3),    dtype=np.float32)
    cov_t = np.zeros((T, N, 3, 3), dtype=np.float32)

    for t in range(T):
        t_off = poses[t, :3]
        R     = quat_xyzw_to_R(poses[t, 3:])
        dR    = R @ R0.T
        dt    = t_off - dR @ t0_off
        mu_t[t]  = (mu0 @ dR.T) + dt[None]
        cov_t[t] = dR @ cov0 @ dR.T

    sh_t      = np.broadcast_to(sh0[None],      (T,) + sh0.shape).copy()
    opacity_t = np.broadcast_to(opacity0[None], (T,) + opacity0.shape).copy()
    scale_t   = np.broadcast_to(scale0[None],   (T,) + scale0.shape).copy()
    return mu_t, cov_t, sh_t, opacity_t, scale_t
