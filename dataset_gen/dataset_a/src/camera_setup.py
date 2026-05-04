"""Object-relative camera placement.

Cameras are positioned in the object's *local* frame and aimed at the
moving-part center, so the action is centered and visible regardless of how
the world places the object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class CameraSpec:
    name: str
    azimuth_deg: float
    elevation_deg: float
    distance_factor: float    # multiplied by bbox_diagonal
    fovy_deg: float
    target: str = "moving_part_center"  # "object_center" | "moving_part_center"


def parse_cameras_yaml(camera_cfg: dict) -> List[CameraSpec]:
    return [CameraSpec(**c) for c in camera_cfg["cameras"]]


def spherical_to_cartesian(azimuth_deg: float,
                            elevation_deg: float,
                            radius: float) -> np.ndarray:
    """Right-handed: x forward (azim=0), y left, z up.

    SAPIEN's world is also z-up by default.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return np.array([x, y, z], dtype=np.float64)


def look_at_pose(camera_pos: np.ndarray,
                 target: np.ndarray,
                 up: np.ndarray = np.array([0, 0, 1.0])):
    """Return SAPIEN Pose with camera looking at target.

    SAPIEN convention for camera pose: x-forward, y-left, z-up.
    """
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation as R

    forward = target - camera_pos
    n = np.linalg.norm(forward)
    if n < 1e-9:
        return sapien.Pose(p=camera_pos.tolist())
    forward = forward / n

    # Make a stable orthonormal basis
    up = np.array(up, dtype=np.float64)
    if abs(np.dot(forward, up)) > 0.999:
        up = np.array([1, 0, 0], dtype=np.float64)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)

    rot = np.eye(3)
    rot[:, 0] = forward
    rot[:, 1] = -right    # SAPIEN: +y is left
    rot[:, 2] = new_up

    quat_xyzw = R.from_matrix(rot).as_quat()
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return sapien.Pose(p=camera_pos.tolist(), q=quat_wxyz)


def add_cameras_to_scene(
    scene,
    specs: List[CameraSpec],
    *,
    object_center: np.ndarray,
    moving_part_center: np.ndarray,
    bbox_diagonal: float,
    image_size: int = 256,
):
    """Add SAPIEN cameras to the scene per the spec list. Returns list of
    (name, sapien_camera) in the same order as `specs`.
    """
    cameras = []
    for spec in specs:
        target = (
            moving_part_center if spec.target == "moving_part_center"
            else object_center
        )
        radius = max(bbox_diagonal * spec.distance_factor, 0.5)
        local = spherical_to_cartesian(
            spec.azimuth_deg, spec.elevation_deg, radius
        )
        cam_pos = target + local
        cam = scene.add_camera(
            name=spec.name,
            width=image_size, height=image_size,
            fovy=np.deg2rad(spec.fovy_deg),
            near=0.05, far=100.0,
        )
        cam.set_pose(look_at_pose(cam_pos, target))
        cameras.append((spec.name, cam))
    return cameras


def get_camera_intrinsics(cam, image_size: int) -> dict:
    """Pinhole intrinsics in OpenCV convention."""
    fovy = float(cam.fovy)  # radians
    fy = image_size / (2 * np.tan(fovy / 2))
    fx = fy   # square pixels
    cx = image_size / 2
    cy = image_size / 2
    return {
        "fx": float(fx), "fy": float(fy),
        "cx": float(cx), "cy": float(cy),
        "width": image_size, "height": image_size,
    }


def camera_extrinsics(cam) -> dict:
    """World-to-camera transform as 4x4 in OpenCV convention.

    SAPIEN's camera pose is camera-to-world in its own (x-forward, y-left, z-up)
    convention. We convert to OpenCV (x-right, y-down, z-forward).
    """
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation as R

    pose = cam.get_pose()
    p = np.array(pose.p)
    quat_wxyz = np.array(pose.q)
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    rot_sap = R.from_quat(quat_xyzw).as_matrix()  # camera-to-world (SAPIEN)

    # SAPIEN camera frame -> OpenCV camera frame
    sap_to_cv = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0],
    ], dtype=np.float64)
    rot_cv = rot_sap @ sap_to_cv

    R_w2c = rot_cv.T
    t_w2c = -R_w2c @ p

    extr = np.eye(4)
    extr[:3, :3] = R_w2c
    extr[:3, 3] = t_w2c
    return {"world_to_camera_4x4": extr.tolist()}
