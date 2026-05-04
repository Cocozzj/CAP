"""Adaptive camera planning per object.

Given a SAPIEN articulation, compute a sensible 3-camera setup that:
  1. targets the AABB center of the whole object (stable across joint states),
  2. distance scales with object size,
  3. pitch (elevation) adapts to object height,
  4. yaw (azimuth) is centered on the direction where the action is most
     visible (motion saliency check between qmin and qmax),
  5. each camera gets random jitter for domain randomization.

This replaces the static cameras.yaml entries with on-the-fly planning per
trajectory. Same `CameraSpec` dataclass as before, so the renderer's
`add_cameras_to_scene` continues to work unchanged.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .camera_setup import CameraSpec

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# AABB
# ----------------------------------------------------------------------------
@dataclass
class ObjectAABB:
    aabb_min: np.ndarray   # (3,) world-space
    aabb_max: np.ndarray   # (3,)
    center: np.ndarray     # (3,)
    diagonal: float        # ||aabb_max - aabb_min||
    height: float          # aabb_max.z - aabb_min.z


def compute_object_aabb(articulation) -> ObjectAABB:
    """Compute world-space AABB by walking all link visual meshes.

    Falls back to a generous default if no meshes are accessible.
    """
    mins = []
    maxs = []
    for link in articulation.get_links():
        try:
            for shape in link.get_collision_shapes():
                # Each shape has a get_local_bounding_box() in SAPIEN 2.x
                if hasattr(shape, "get_local_bounding_box"):
                    aabb = shape.get_local_bounding_box()
                    if aabb is not None:
                        local_min = np.asarray(aabb[0])
                        local_max = np.asarray(aabb[1])
                        # Transform to world via link pose
                        link_pose = link.get_pose()
                        # SAPIEN Pose has p (position) and q (quaternion wxyz)
                        # For AABB transform we use the 8 corners
                        corners_local = np.array([
                            [local_min[0], local_min[1], local_min[2]],
                            [local_max[0], local_min[1], local_min[2]],
                            [local_min[0], local_max[1], local_min[2]],
                            [local_max[0], local_max[1], local_min[2]],
                            [local_min[0], local_min[1], local_max[2]],
                            [local_max[0], local_min[1], local_max[2]],
                            [local_min[0], local_max[1], local_max[2]],
                            [local_max[0], local_max[1], local_max[2]],
                        ])
                        from scipy.spatial.transform import Rotation as R
                        rot = R.from_quat([link_pose.q[1], link_pose.q[2],
                                            link_pose.q[3], link_pose.q[0]]).as_matrix()
                        corners_world = corners_local @ rot.T + np.asarray(link_pose.p)
                        mins.append(corners_world.min(axis=0))
                        maxs.append(corners_world.max(axis=0))
        except Exception:
            # Fall back to link pose
            try:
                link_pose = link.get_pose()
                p = np.asarray(link_pose.p)
                mins.append(p - 0.1)
                maxs.append(p + 0.1)
            except Exception:
                continue

    if not mins:
        # Couldn't get any AABB info; use generous default
        center = np.zeros(3)
        return ObjectAABB(
            aabb_min=center - 1.0, aabb_max=center + 1.0,
            center=center, diagonal=2.0 * np.sqrt(3), height=2.0,
        )

    aabb_min = np.array(mins).min(axis=0)
    aabb_max = np.array(maxs).max(axis=0)
    center = 0.5 * (aabb_min + aabb_max)
    extent = aabb_max - aabb_min
    diagonal = float(np.linalg.norm(extent))
    height = float(extent[2])
    return ObjectAABB(
        aabb_min=aabb_min, aabb_max=aabb_max,
        center=center, diagonal=diagonal, height=height,
    )


# ----------------------------------------------------------------------------
# best yaw via motion saliency
# ----------------------------------------------------------------------------
def find_best_yaw(
    scene,
    articulation,
    target_joint_idx: int,
    qpos_low: float,
    qpos_high: float,
    aabb: ObjectAABB,
    *,
    candidates_deg: List[float] = (0.0, 90.0, 180.0, 270.0),
    image_size: int = 64,
    fovy_deg: float = 45.0,
    distance_factor: float = 1.5,
    elevation_deg: float = 25.0,
) -> Tuple[float, dict]:
    """For each candidate yaw, render the object at qmin and qmax, compute the
    mean pixel diff, and return the yaw that maximizes motion visibility.

    Returns (best_yaw_deg, {yaw -> pixel_diff_score}).
    """
    import sapien.core as sapien
    from .camera_setup import look_at_pose, spherical_to_cartesian

    # Add a temporary camera; we'll move it for each candidate
    cam = scene.add_camera(
        name="best_yaw_probe",
        width=image_size, height=image_size,
        fovy=np.deg2rad(fovy_deg),
        near=0.05, far=100.0,
    )

    # Save original qpos so we can restore
    active_joints = articulation.get_active_joints()
    qpos_full = np.array(articulation.get_qpos(), dtype=np.float64).copy()
    radius = max(aabb.diagonal * distance_factor, 0.5)

    scores: dict = {}
    for yaw in candidates_deg:
        # Move cam to (yaw, elevation, distance) relative to AABB center
        local = spherical_to_cartesian(yaw, elevation_deg, radius)
        cam_pos = aabb.center + local
        cam.set_pose(look_at_pose(cam_pos, aabb.center))

        # qmin
        qpos_full[target_joint_idx] = float(qpos_low)
        articulation.set_qpos(qpos_full)
        scene.update_render()
        cam.take_picture()
        rgb_min = (cam.get_color_rgba()[..., :3]).copy()

        # qmax
        qpos_full[target_joint_idx] = float(qpos_high)
        articulation.set_qpos(qpos_full)
        scene.update_render()
        cam.take_picture()
        rgb_max = (cam.get_color_rgba()[..., :3]).copy()

        diff = float(np.abs(rgb_min - rgb_max).mean())
        scores[float(yaw)] = diff

    # Remove probe camera
    try:
        scene.remove_camera(cam)
    except Exception:
        pass

    best_yaw = max(scores.items(), key=lambda x: x[1])[0]
    logger.info("best_yaw scores: %s -> %.1f deg",
                {k: round(v, 4) for k, v in scores.items()}, best_yaw)
    return best_yaw, scores


# ----------------------------------------------------------------------------
# main planner
# ----------------------------------------------------------------------------
def plan_cameras(
    aabb: ObjectAABB,
    best_yaw_deg: float,
    rng: random.Random,
    cfg: dict,
) -> Tuple[List[CameraSpec], np.ndarray]:
    """Build 3 CameraSpec entries with adaptive distance/pitch + jitter.

    `cfg` is the `camera_design` block of cameras.yaml. Returns
    (specs, target_world_xyz). The target xyz is the AABB center; the
    renderer should pass this as `object_center` to `add_cameras_to_scene`,
    and cameras with target='object_center' will look at it.
    """
    # Distance
    base_dist_factor = cfg.get("distance_factor", 1.5)
    dist_jitter_lo, dist_jitter_hi = cfg.get("distance_jitter", [0.85, 1.15])
    dist_ratio = rng.uniform(dist_jitter_lo, dist_jitter_hi)
    distance_factor = base_dist_factor * dist_ratio

    # Pitch (elevation, in our convention; positive = above target)
    height_threshold = cfg.get("height_threshold", 0.8)
    if aabb.height > height_threshold:
        base_pitch = cfg.get("pitch_tall_object", 25.0)   # +25 above horizontal
    else:
        base_pitch = cfg.get("pitch_short_object", 35.0)
    pitch_jitter_lo, pitch_jitter_hi = cfg.get("pitch_jitter_deg", [-10, 10])
    pitch_offset = rng.uniform(pitch_jitter_lo, pitch_jitter_hi)
    base_elevation = base_pitch + pitch_offset

    # Yaw
    yaw_spread = cfg.get("yaw_spread_deg", 60.0)
    yaw_jitter_lo, yaw_jitter_hi = cfg.get("yaw_jitter_deg", [-20, 20])
    yaw_offset_global = rng.uniform(yaw_jitter_lo, yaw_jitter_hi)

    fovy = cfg.get("fovy_deg", 45.0)

    # Three cameras: best_yaw + offset, best_yaw + offset + spread, best_yaw + offset - spread
    yaws = [
        best_yaw_deg + yaw_offset_global,
        best_yaw_deg + yaw_offset_global + yaw_spread,
        best_yaw_deg + yaw_offset_global - yaw_spread,
    ]

    specs: List[CameraSpec] = []
    for i, yaw in enumerate(yaws):
        specs.append(CameraSpec(
            name=f"cam{i}",
            azimuth_deg=float(yaw % 360),
            elevation_deg=float(base_elevation),
            distance_factor=float(distance_factor),
            fovy_deg=float(fovy),
            target="object_center",
        ))

    return specs, aabb.center


# ----------------------------------------------------------------------------
# top-level helper that does it all
# ----------------------------------------------------------------------------
def plan_cameras_for_articulation(
    scene,
    articulation,
    target_joint_idx: int,
    qpos_low: float,
    qpos_high: float,
    cfg: dict,
    seed: int = 0,
) -> Tuple[List[CameraSpec], np.ndarray, ObjectAABB]:
    """Convenience: compute AABB, find best yaw, plan cameras.

    Caller can cache (best_yaw, AABB) per object_id and reuse across
    trajectories of the same object — only the random jitter changes per
    trajectory.

    Returns (specs, target_world_xyz, aabb).
    """
    aabb = compute_object_aabb(articulation)

    # If best_yaw motion check disabled (e.g. for soft / no-joint objects), use 0
    if cfg.get("best_yaw_motion_check", True):
        candidates = cfg.get("best_yaw_candidates", [0, 90, 180, 270])
        try:
            best_yaw, _ = find_best_yaw(
                scene, articulation,
                target_joint_idx=target_joint_idx,
                qpos_low=qpos_low, qpos_high=qpos_high,
                aabb=aabb,
                candidates_deg=candidates,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("find_best_yaw failed (%s); fallback to yaw=0", e)
            best_yaw = 0.0
    else:
        best_yaw = 0.0

    rng = random.Random(seed)
    specs, target_xyz = plan_cameras(aabb, best_yaw, rng, cfg)
    return specs, target_xyz, aabb
