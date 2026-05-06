"""Extract the moving-part trajectory from a predicted 4DGS sequence.

For evaluating ADE/FDE/MPJPE/Success at the *part* level, we need:

  • The trajectory of the **moving part's centroid** in pred — analogous
    to the GT joint angle trajectory.
  • Optionally, a few **keypoint** trajectories (e.g. corners of the
    moving part's bounding box) for MPJPE.

We don't have per-Gaussian link assignment, so we use motion analysis:
points whose temporal motion magnitude ranks in the top-K% are treated
as "moving part".  This works for:

  • TAMP / MotionGPT / Ours — pred has T frames, motion ranking is direct.
  • PhysGaussian — same.
  • Static predictions (Ours T=1, MotionGPT mode-collapse) — motion ≈ 0
    everywhere; we fall back to using all Gaussians' centroid.

Note on Ours T=1: when the model emits a single-step trajectory, we treat
it as "init → final" with linear interpolation for the metric path; this
makes the ADE/FDE comparable but reflects the model's coarse temporal
output (paper notes this honestly in the methods section).
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def estimate_moving_mask(
    mu:        np.ndarray,         # [T, N, 3]
    top_frac:  float = 0.30,
    eps:       float = 1e-6,
) -> np.ndarray:
    """Return a [N] boolean mask: True for Gaussians in the moving part.

    Heuristic: motion magnitude per-Gaussian = ‖mu(T-1) − mu(0)‖.
    Top ``top_frac`` are flagged.  Falls back to all-True if total
    motion is below ``eps`` (truly static prediction).
    """
    if mu.shape[0] < 2:
        return np.ones(mu.shape[1], dtype=bool)
    motion = np.linalg.norm(mu[-1] - mu[0], axis=-1)            # [N]
    if motion.max() < eps:
        return np.ones(mu.shape[1], dtype=bool)
    K = max(int(top_frac * motion.shape[0]), 1)
    idx = np.argpartition(-motion, kth=K - 1)[:K]
    mask = np.zeros(motion.shape[0], dtype=bool)
    mask[idx] = True
    return mask


def part_centroid_trajectory(
    mu:        np.ndarray,         # [T, N, 3]
    mask:      np.ndarray,         # [N] bool
) -> np.ndarray:
    """Centroid of masked Gaussians at every t.  Shape [T, 3]."""
    if not mask.any():
        return mu.mean(axis=1)                                  # all-fallback
    sub = mu[:, mask]                                           # [T, n_moving, 3]
    return sub.mean(axis=1)


def part_keypoints_indices(
    mu:        np.ndarray,         # [T, N, 3]
    mask:      np.ndarray,         # [N] bool
    n_kpts:    int = 4,
) -> np.ndarray:
    """Compute the n_kpts Gaussian indices (corners of moving part's AABB)."""
    if not mask.any():
        # All-fallback: 4 spatial-extreme Gaussians of the whole cloud
        order = mu[0].std(axis=-1).argsort()[::-1][:n_kpts]
        return np.asarray(order, dtype=np.int64)
    moving_idx = np.where(mask)[0]
    sub0 = mu[0, moving_idx]                                    # [n_mov, 3]
    # Pick 4 corners by extremes of (x+y+z), (x-y+z), (-x+y+z), (-x-y+z)
    signs = np.array([
        [+1, +1, +1],
        [+1, -1, +1],
        [-1, +1, +1],
        [-1, -1, +1],
    ], dtype=np.float32)[: n_kpts]
    order = []
    for s in signs:
        scores = sub0 @ s
        order.append(int(moving_idx[int(scores.argmax())]))
    return np.asarray(order, dtype=np.int64)


def part_keypoints(
    mu:        np.ndarray,         # [T, N, 3]
    mask:      np.ndarray,         # [N] bool
    n_kpts:    int = 4,
) -> np.ndarray:
    """Extract ``n_kpts`` representative keypoints on the moving part.

    Strategy: at frame 0, take the moving-Gaussians' AABB corners
    (typically 4–8 corners, default 4 = mid-X face corners), then track
    those *same Gaussian indices* across frames.  This gives a stable
    correspondence between pred and GT keypoint trajectories — both use
    AABB corner Gaussians.

    Returns [T, n_kpts, 3].
    """
    order = part_keypoints_indices(mu, mask, n_kpts=n_kpts)
    return mu[:, order]                                          # [T, n_kpts, 3]


def estimate_pred_joint_angle_trajectory(
    centroid_traj: np.ndarray,     # [T, 3] pred moving-part centroid in world frame
    joint_origin:  np.ndarray,     # [3]    URDF joint origin in world (= obj root)
    joint_axis:    np.ndarray,     # [3]    URDF joint axis (unit)
) -> np.ndarray:
    """Recover an angle-around-axis trajectory from a centroid trajectory.

    For revolute joints: the centroid of the moving link traces a circular
    arc as the joint rotates.  Project (C(t) − origin) onto the plane
    perpendicular to the joint axis and measure the signed angle from
    (C(0) − origin)'s in-plane component.
    """
    a = joint_axis / max(float(np.linalg.norm(joint_axis)), 1e-12)
    r0 = centroid_traj[0] - joint_origin
    # Project out the component along the axis (so r0_perp lies in the plane)
    r0_perp = r0 - np.dot(r0, a) * a
    norm_r0 = max(float(np.linalg.norm(r0_perp)), 1e-12)
    e1 = r0_perp / norm_r0
    # e2 is in-plane and perpendicular to e1, sign aligned with axis
    e2 = np.cross(a, e1)

    angles = np.zeros(centroid_traj.shape[0], dtype=np.float32)
    for t in range(centroid_traj.shape[0]):
        rt = centroid_traj[t] - joint_origin
        rt_perp = rt - np.dot(rt, a) * a
        x = float(np.dot(rt_perp, e1))
        y = float(np.dot(rt_perp, e2))
        angles[t] = float(np.arctan2(y, x))
    return angles


def estimate_pred_displacement_trajectory(
    centroid_traj: np.ndarray,     # [T, 3]
    joint_axis:    np.ndarray,     # [3]
) -> np.ndarray:
    """For prismatic joints: signed displacement of centroid along axis."""
    a = joint_axis / max(float(np.linalg.norm(joint_axis)), 1e-12)
    return ((centroid_traj - centroid_traj[0]) @ a).astype(np.float32)


__all__ = [
    "estimate_moving_mask",
    "part_centroid_trajectory",
    "part_keypoints",
    "estimate_pred_joint_angle_trajectory",
    "estimate_pred_displacement_trajectory",
]
