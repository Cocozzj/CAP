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
    part_keypoints_indices,
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

    # Ours' planner emits a single macro-step → pred_mu has T=1.  In that
    # case, pred[0] is interpreted as the *final* scene after the action
    # (start = init_gs.ply, end = pred[0]).  Expand to a trivial 2-frame
    # trajectory so ADE/FDE compare end-to-end rather than self-to-self.
    if T_pred == 1:
        from .aggregate import _load_init_mu
        init_world = _load_init_mu(traj_dir, n_points=int(pred_mu.shape[1]))
        if init_world is not None:
            init_centroid_world = init_world[mask].mean(axis=0) if mask.any() \
                                  else init_world.mean(axis=0)
            init_kpts_world = init_world[part_keypoints_indices(
                init_world[None], mask, n_kpts=n_kpts,
            )]
            pred_centroid = np.stack([init_centroid_world, pred_centroid[0]], axis=0)
            pred_kpts = np.stack([init_kpts_world, pred_kpts[0]], axis=0)
            T_pred = 2

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

    # ── Success: tighter criterion that requires BOTH magnitude AND
    #            DIRECTION alignment between pred and GT centroid motion.
    #            (The previous version checked only |Δ| magnitude, which
    #            allowed PhysGaussian's random-direction MPM blob motion to
    #            pass when its displacement happened to match GT in size.)
    #
    # Tightened criteria:
    #   1. Pred centroid displacement vector must align with GT centroid
    #      displacement vector (cos similarity ≥ 0.5, i.e., within ~60°).
    #   2. For revolute: pred *signed* joint angle change must have the
    #      same sign as GT and reach ≥ ``angle_rad`` magnitude.
    #   3. For prismatic: pred *signed* displacement along axis must match
    #      sign of GT change and reach ≥ ``distance_m``.
    th = threshold_for(task_name)
    if th is not None:
        # Direction agreement on centroid displacement vectors (works for
        # both revolute and prismatic — the centroid moves either along
        # an arc or along the axis, both have a clear direction).
        pred_disp_vec = pred_centroid[-1] - pred_centroid[0]   # [3]
        gt_disp_vec   = gt_centroid[-1]   - gt_centroid[0]
        pred_mag = float(np.linalg.norm(pred_disp_vec))
        gt_mag   = float(np.linalg.norm(gt_disp_vec))
        if pred_mag > 1e-6 and gt_mag > 1e-6:
            cos_dir = float(pred_disp_vec @ gt_disp_vec / (pred_mag * gt_mag))
        else:
            cos_dir = 0.0
        ok_direction = (cos_dir >= 0.5)        # within ~60° of GT direction

        if is_revolute_task(task_name) and j.type in ("revolute", "continuous"):
            pred_angles = estimate_pred_joint_angle_trajectory(
                pred_centroid, j.origin_xyz, j.axis,
            )
            pred_change_signed = float(pred_angles[-1] - pred_angles[0])
            gt_change_signed   = float(joint_qpos[-1] - joint_qpos[0])
            # Sign match (both need to articulate in the SAME direction)
            ok_sign = (
                gt_change_signed * pred_change_signed > 0
                or abs(gt_change_signed) < 0.01      # GT didn't really move → skip
            )
            ok_magnitude = abs(pred_change_signed) >= float(th.angle_rad or 0.0)
            # Tolerance: pred magnitude within 50% of GT magnitude (looser
            # than direction; we mostly care about getting close to GT goal)
            if abs(gt_change_signed) > 1e-3:
                rel = abs(abs(pred_change_signed) - abs(gt_change_signed)) \
                      / abs(gt_change_signed)
                ok_magnitude = ok_magnitude and (rel <= 0.50)
            out["success"] = float(ok_sign and ok_magnitude and ok_direction)

        elif j.type == "prismatic":
            pred_disp_signed = estimate_pred_displacement_trajectory(
                pred_centroid, j.axis,
            )
            pred_change_signed = float(pred_disp_signed[-1] - pred_disp_signed[0])
            gt_change_signed   = float(joint_qpos[-1] - joint_qpos[0])
            ok_sign = (
                gt_change_signed * pred_change_signed > 0
                or abs(gt_change_signed) < 1e-3
            )
            ok_magnitude = abs(pred_change_signed) >= float(th.distance_m or 0.0)
            if abs(gt_change_signed) > 1e-4:
                rel = abs(abs(pred_change_signed) - abs(gt_change_signed)) \
                      / abs(gt_change_signed)
                ok_magnitude = ok_magnitude and (rel <= 0.50)
            out["success"] = float(ok_sign and ok_magnitude and ok_direction)

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
