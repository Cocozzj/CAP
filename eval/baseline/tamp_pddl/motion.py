"""Motion primitive library for the TAMP baseline.

Each PDDL action (open / close / push / pull / rotate / ...) maps to a
trajectory generator here.  Output is a per-frame object_pose_world
[T, 7] = (xyz + xyzw quaternion) that gets applied to init_gs.

The trajectories use the GT pose endpoints from trajectory.npz when
available (fair: TAMP has access to pose oracles), else fall back to
linear motion along a default axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Endpoint loading
# ──────────────────────────────────────────────────────────────────────

@dataclass
class PoseEndpoints:
    pose0: np.ndarray   # [7]
    poseT: np.ndarray   # [7]


def load_pose_endpoints(traj_dir: Path | str) -> Optional[PoseEndpoints]:
    """Read object_pose_world[0] and [-1] from trajectory.npz."""
    p = Path(traj_dir) / "trajectory.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "object_pose_world" not in z.files:
        return None
    poses = z["object_pose_world"].astype(np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 2:
        return None
    return PoseEndpoints(pose0=poses[0], poseT=poses[-1])


# ──────────────────────────────────────────────────────────────────────
# Quaternion utilities (xyzw convention)
# ──────────────────────────────────────────────────────────────────────

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / max(n, 1e-12)


def quat_slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = quat_normalize(q0); q1 = quat_normalize(q1)
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        return quat_normalize((1 - u) * q0 + u * q1)
    theta_0 = float(np.arccos(dot))
    theta   = theta_0 * u
    sin0    = float(np.sin(theta_0))
    s0      = float(np.cos(theta) - dot * np.sin(theta) / sin0)
    s1      = float(np.sin(theta) / sin0)
    return (s0 * q0 + s1 * q1).astype(np.float32)


def axis_angle_to_quat(axis: np.ndarray, angle: float) -> np.ndarray:
    """xyzw quaternion for rotation by `angle` around unit `axis`."""
    a = quat_normalize(np.append(axis, 0.0))[:3]   # ensure unit axis
    s = float(np.sin(angle / 2)); c = float(np.cos(angle / 2))
    return np.array([a[0]*s, a[1]*s, a[2]*s, c], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Motion primitives
# ──────────────────────────────────────────────────────────────────────

def linear_pose_trajectory(p0: np.ndarray, pT: np.ndarray, T: int) -> np.ndarray:
    """Linear interpolation in translation + SLERP in rotation."""
    out = np.zeros((T, 7), dtype=np.float32)
    for i in range(T):
        u = i / max(T - 1, 1)
        out[i, :3] = (1 - u) * p0[:3] + u * pT[:3]
        out[i, 3:] = quat_slerp(p0[3:], pT[3:], u)
    return out


def linear_translate(p0: np.ndarray, direction: np.ndarray,
                       distance: float, T: int) -> np.ndarray:
    """Translate from p0 by ``distance × direction`` over T frames (no rotation)."""
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    out = np.tile(p0.copy(), (T, 1)).astype(np.float32)
    for i in range(T):
        u = i / max(T - 1, 1)
        out[i, :3] = p0[:3] + u * distance * direction
    return out


def rotate_around_axis(p0: np.ndarray, axis: np.ndarray, angle: float,
                          T: int) -> np.ndarray:
    """Rotate from p0 by `angle` around axis over T frames (no translation)."""
    out = np.tile(p0.copy(), (T, 1)).astype(np.float32)
    base = quat_normalize(p0[3:])
    for i in range(T):
        u = i / max(T - 1, 1)
        dq = axis_angle_to_quat(axis, angle * u)
        # Apply dq * base
        x1, y1, z1, w1 = dq
        x2, y2, z2, w2 = base
        out[i, 3:] = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float32)
    return out


# ──────────────────────────────────────────────────────────────────────
# Action → trajectory dispatcher
# ──────────────────────────────────────────────────────────────────────

# Default joint axes (used as fallback when GT not available).  Most
# PartNet-Mobility hinges lie close to z-axis.
_DEFAULT_HINGE_AXIS = np.array([0, 0, 1], dtype=np.float32)
_DEFAULT_PUSH_DIR   = np.array([1, 0, 0], dtype=np.float32)


def execute_action(
    action_name:  str,
    traj_dir:     Path,
    T:            int,
    push_distance: float = 0.3,
    rotate_angle:  float = np.pi / 2,
) -> Optional[np.ndarray]:
    """Generate a [T, 7] trajectory for one PDDL action.

    Strategy:
      - If trajectory.npz endpoints available → linear LERP+SLERP between them
        (fair use of "pose oracle" by TAMP)
      - Else fall back to a heuristic primitive based on action_name

    Returns None if the action has no defined motion primitive (e.g.
    soft-body verbs fold/squeeze/pour).
    """
    name = action_name.lower()

    # Soft-body verbs: TAMP cannot execute these (no deformation model)
    if name in ("fold", "squeeze", "pour"):
        return None

    endpoints = load_pose_endpoints(traj_dir)

    # Open / Close / Rotate: use GT endpoints if available
    if name in ("open", "close", "rotate"):
        if endpoints is not None:
            if name == "close":
                # Reverse the trajectory direction for "close"
                return linear_pose_trajectory(endpoints.poseT, endpoints.pose0, T)
            return linear_pose_trajectory(endpoints.pose0, endpoints.poseT, T)
        # Fallback: rotate around z by 90deg
        if endpoints is None:
            base = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
            return rotate_around_axis(
                base, _DEFAULT_HINGE_AXIS,
                rotate_angle if name != "close" else -rotate_angle, T,
            )

    # Push / Pull: linear translate
    if name in ("push", "pull"):
        if endpoints is not None:
            if name == "pull":
                return linear_pose_trajectory(endpoints.poseT, endpoints.pose0, T)
            return linear_pose_trajectory(endpoints.pose0, endpoints.poseT, T)
        base = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        sign = 1.0 if name == "push" else -1.0
        return linear_translate(base, _DEFAULT_PUSH_DIR, sign * push_distance, T)

    # Unknown verb
    return None


def chain_actions(
    plan_actions: list[str],
    traj_dir:     Path,
    T:            int,
) -> Optional[np.ndarray]:
    """Concatenate trajectories for a multi-step plan (e.g. "comp:close_open").

    Each action gets ⌈T / N⌉ frames.  Inter-step state continuity is
    preserved by setting each next action's start pose to the previous
    action's end pose.
    """
    if not plan_actions:
        return None
    N = len(plan_actions)
    T_per = max(2, T // N)
    out: list[np.ndarray] = []
    cur_pose: Optional[np.ndarray] = None
    for action in plan_actions:
        seg = execute_action(action, traj_dir, T=T_per)
        if seg is None:
            return None
        if cur_pose is not None:
            # Smooth shift so seg[0] == cur_pose
            shift_t = cur_pose[:3] - seg[0, :3]
            seg[:, :3] += shift_t[None]
        out.append(seg)
        cur_pose = seg[-1].copy()

    poses = np.concatenate(out, axis=0)
    # Pad / truncate to exactly T frames
    if poses.shape[0] < T:
        pad = np.tile(poses[-1:], (T - poses.shape[0], 1))
        poses = np.concatenate([poses, pad], axis=0)
    return poses[:T]
