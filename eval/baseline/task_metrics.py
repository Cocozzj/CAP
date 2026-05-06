"""Task-aware ADE / FDE / MPJPE / Success metrics.

Replaces the old "compare all 10000 Gaussian centers vs static GT object
pose" formulation in ``metrics.py`` (which was meaningless for articulated
PartNet objects, where ``object_pose_world`` is constant and the real
motion lives in ``joint_qpos``).

Per the paper spec:
  • ADE — average trajectory deviation of the **target part** (e.g. door
    angle delta over time).
  • FDE — final state deviation.
  • MPJPE — average over multiple keypoints on the part (4 corners).
  • Success — task-specific threshold (open ≥ 30°, push ≥ 5 cm, …).

Implementation outline (per-trajectory):

  1. Read ``meta.json`` → identify ``joint_index`` + ``task_name``.
  2. Read ``trajectory.npz["joint_qpos"]`` → 1-D GT joint trajectory.
  3. Read ``mobility.urdf`` → joint axis / origin / type.
  4. From pred 4DGS:
       a. moving-part mask (top-K motion Gaussians)
       b. centroid trajectory  → ADE (Euclidean)
       c. recover joint angle trajectory  → success
       d. AABB-corner keypoint trajectory → MPJPE
  5. From GT:
       a. joint_qpos[t] is the GT joint scalar
       b. compose with init Gaussian centroid + URDF FK to get GT
          centroid trajectory  → ADE reference
       c. apply FK to AABB corners → GT MPJPE reference

Returns a dict that fills the ``ade / fde / mpjpe / success`` slots in
``TrajMetrics``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .keypoint_extractor import (
    estimate_moving_mask,
    estimate_pred_displacement_trajectory,
    estimate_pred_joint_angle_trajectory,
    part_centroid_trajectory,
    part_keypoints,
)
from .success_thresholds import (
    is_revolute_task,
    threshold_for,
)
from .urdf_fk import (
    JointInfo,
    get_joint_by_index,
    joint_pose,
    load_partnet_urdf,
    rotation_matrix,
)


# ════════════════════════════════════════════════════════════════════════
# GT trajectory reconstruction (using URDF + joint_qpos)
# ════════════════════════════════════════════════════════════════════════

def _gt_centroid_trajectory(
    init_centroid: np.ndarray,     # [3] centroid of moving Gaussians at t=0 in world
    joint_qpos:    np.ndarray,     # [T_gt] scalar joint trajectory
    j:             JointInfo,
    T_pred:        int,
) -> np.ndarray:
    """Apply URDF FK to the (single) moving link's centroid for each
    target joint angle along the trajectory.  Returns [T_pred, 3].

    For revolute: rotate (centroid − origin) by Δθ = q(t) − q(0) around
    axis, then add origin back.
    For prismatic: translate centroid by (q(t) − q(0)) · axis.
    """
    T_gt = joint_qpos.shape[0]
    idx = np.linspace(0, T_gt - 1, T_pred, dtype=int)
    out = np.zeros((T_pred, 3), dtype=np.float32)
    q0 = float(joint_qpos[0])
    for i, fi in enumerate(idx):
        dq = float(joint_qpos[int(fi)]) - q0
        if j.type in ("revolute", "continuous"):
            R = rotation_matrix(j.axis, dq)
            r = init_centroid - j.origin_xyz
            out[i] = R @ r + j.origin_xyz
        elif j.type == "prismatic":
            a = j.axis / max(float(np.linalg.norm(j.axis)), 1e-12)
            out[i] = init_centroid + dq * a
        else:
            out[i] = init_centroid
    return out


def _gt_keypoint_trajectory(
    init_keypoints: np.ndarray,    # [n_kpts, 3] in world frame
    joint_qpos:     np.ndarray,
    j:              JointInfo,
    T_pred:         int,
) -> np.ndarray:
    """Per-keypoint version of _gt_centroid_trajectory.  Shape [T_pred, n_kpts, 3]."""
    T_gt = joint_qpos.shape[0]
    idx = np.linspace(0, T_gt - 1, T_pred, dtype=int)
    n = init_keypoints.shape[0]
    out = np.zeros((T_pred, n, 3), dtype=np.float32)
    q0 = float(joint_qpos[0])
    for i, fi in enumerate(idx):
        dq = float(joint_qpos[int(fi)]) - q0
        if j.type in ("revolute", "continuous"):
            R = rotation_matrix(j.axis, dq)
            r = init_keypoints - j.origin_xyz
            out[i] = (r @ R.T) + j.origin_xyz
        elif j.type == "prismatic":
            a = j.axis / max(float(np.linalg.norm(j.axis)), 1e-12)
            out[i] = init_keypoints + dq * a
        else:
            out[i] = init_keypoints
    return out


# ════════════════════════════════════════════════════════════════════════
# Public entry point
# ════════════════════════════════════════════════════════════════════════

def compute_task_metrics(
    pred_mu:    np.ndarray,         # [T_pred, N, 3]
    traj_dir:   Path,
    partnet_raw_dir: Optional[Path] = None,
    top_frac:   float = 0.30,
    n_kpts:     int   = 4,
) -> Dict[str, Optional[float]]:
    """Compute task-aware ADE / FDE / MPJPE / Success from pred + GT.

    Returns a dict with keys ``ade, fde, mpjpe, success`` (all floats or
    ``None`` when not computable).  ``ade/fde`` are in metres, MPJPE same
    units, success ∈ {0.0, 1.0}.
    """
    out: Dict[str, Optional[float]] = {
        "ade": None, "fde": None, "mpjpe": None, "success": None,
    }

    # ── Load meta + trajectory + URDF ──
    meta_path = Path(traj_dir) / "meta.json"
    traj_path = Path(traj_dir) / "trajectory.npz"
    if not meta_path.exists() or not traj_path.exists():
        return out
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return out

    obj_id    = str(meta.get("obj_id", ""))
    joint_idx = int(meta.get("joint_index", 0))
    task_name = str(meta.get("task_name", ""))
    if not obj_id:
        return out

    joints = load_partnet_urdf(
        obj_id, partnet_raw_dir if partnet_raw_dir else _default_partnet_dir(meta),
    )
    if joints is None:
        return out
    j = get_joint_by_index(joints, joint_idx)
    if j is None or j.type not in ("revolute", "continuous", "prismatic"):
        return out

    try:
        z = np.load(traj_path, allow_pickle=False)
        joint_qpos = np.asarray(z["joint_qpos"], dtype=np.float32)
    except Exception:
        return out
    if joint_qpos.ndim != 1 or joint_qpos.shape[0] < 2:
        return out

    T_pred = int(pred_mu.shape[0])

    # ── Pred side: identify moving Gaussians + their centroid + keypoints ──
    mask = estimate_moving_mask(pred_mu, top_frac=top_frac)
    pred_centroid = part_centroid_trajectory(pred_mu, mask)         # [T_pred, 3]
    pred_kpts     = part_keypoints(pred_mu, mask, n_kpts=n_kpts)    # [T_pred, K, 3]

    # ── GT side: same shape, computed by FK ──
    init_centroid = pred_centroid[0]
    init_kpts     = pred_kpts[0]
    gt_centroid = _gt_centroid_trajectory(
        init_centroid, joint_qpos, j, T_pred,
    )
    gt_kpts = _gt_keypoint_trajectory(
        init_kpts, joint_qpos, j, T_pred,
    )

    # ── Geometric metrics ──
    err = np.linalg.norm(pred_centroid - gt_centroid, axis=-1)      # [T_pred]
    out["ade"] = float(err.mean())
    out["fde"] = float(err[-1])
    err_kp = np.linalg.norm(pred_kpts - gt_kpts, axis=-1)           # [T_pred, K]
    out["mpjpe"] = float(err_kp.mean())

    # ── Success: did the moving part travel sufficiently in the
    #            correct direction (revolute) or along the axis
    #            (prismatic)?  Compare to GT total magnitude. ──
    th = threshold_for(task_name)
    if th is not None:
        if is_revolute_task(task_name) and j.type in ("revolute", "continuous"):
            pred_angles = estimate_pred_joint_angle_trajectory(
                pred_centroid, j.origin_xyz, j.axis,
            )
            pred_change = abs(float(pred_angles[-1] - pred_angles[0]))
            gt_change   = abs(float(joint_qpos[-1] - joint_qpos[0]))
            ok = pred_change >= float(th.angle_rad or 0.0)
            # Optional: require pred-to-GT alignment within tolerance_frac
            if gt_change > 1e-3:
                rel = abs(pred_change - gt_change) / gt_change
                ok = ok and (rel <= max(th.tolerance_frac, 0.30))
            out["success"] = float(ok)
        elif j.type == "prismatic":
            pred_disp = estimate_pred_displacement_trajectory(
                pred_centroid, j.axis,
            )
            pred_change = abs(float(pred_disp[-1] - pred_disp[0]))
            ok = pred_change >= float(th.distance_m or 0.0)
            out["success"] = float(ok)

    return out


# ────────────────────────────────────────────────────────────────────────

def _default_partnet_dir(meta: dict) -> str:
    """Pick the PartNet raw dir from meta.json's ``obj_folder`` (if absolute)
    or fall back to the cluster default."""
    folder = meta.get("obj_folder", "")
    if folder and Path(folder).exists():
        return str(Path(folder).parent)
    # fallback
    return "/home/zejun/CAP/dataset_gen/raw_data/partnet-mobility/dataset"


__all__ = ["compute_task_metrics"]
