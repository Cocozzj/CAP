"""Pyrender-based renderer for soft-body trajectories.

The articulated renderer (renderer.py) uses SAPIEN to replay joint motion.
Soft trajectories instead carry per-frame deformation parameters (scale,
fold angle, etc.); we apply them to the rest mesh from the soft_object_spec
and rasterize with pyrender.

Outputs are written in the same format as renderer.py:
    <out_dir>/{front,side,high_oblique}.mp4
            + cameras.json (intrinsics + world-to-camera 4x4)
            + trajectory.npz (just records frame indices for soft tasks)
            + physics.json
            + meta.json (with object_type='soft')

Camera placement uses the same CameraSpec list as the SAPIEN renderer, so the
3 views are spatially consistent with articulated trajectories.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .camera_setup import CameraSpec
from .soft_objects import (
    SoftObjectSpec,
    apply_anisotropic_scale,
    apply_hinge_fold,
    make_from_spec,
)

logger = logging.getLogger(__name__)


@dataclass
class SoftRenderResult:
    traj_id: str
    out_dir: Path
    n_frames_written: int
    cameras_meta: dict


# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------
def render_soft_trajectory(
    traj_record: dict,
    camera_specs: List[CameraSpec],
    *,
    image_size: int = 256,
    out_dir: str | Path,
    fps: int = 30,
) -> Optional[SoftRenderResult]:
    """Render a soft-object trajectory to MP4s using pyrender."""
    try:
        import pyrender
        import imageio
        import trimesh
    except ImportError as e:
        logger.error("pyrender / trimesh / imageio not installed: %s", e)
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = SoftObjectSpec.from_dict(traj_record["soft_object_spec"])
    rest_mesh = make_from_spec(spec)

    deform_seq = traj_record["deformation_params_per_frame"]
    n_frames = len(deform_seq)

    # ---- pyrender scene + lighting (rebuilt per frame because mesh changes)
    cam_nodes_per_view, intr_per_view = _build_camera_nodes(
        camera_specs, image_size=image_size, bbox_diag=_estimate_diag(rest_mesh),
    )

    # Open MP4 writers
    writers = {}
    for spec_cam in camera_specs:
        path = out_dir / f"{spec_cam.name}.mp4"
        writers[spec_cam.name] = imageio.get_writer(
            str(path), fps=fps, codec="libx264",
            macro_block_size=1, quality=8,
        )

    # ---- per-frame render loop
    renderer = pyrender.OffscreenRenderer(image_size, image_size)
    task_name = traj_record["task_name"]

    try:
        for f_idx, deform in enumerate(deform_seq):
            # 1. Apply deformation to rest mesh
            deformed = _apply_deformation(rest_mesh, task_name, deform)

            # 2. Build pyrender scene
            scene = pyrender.Scene(
                bg_color=[0.95, 0.95, 0.95, 1.0],
                ambient_light=[0.3, 0.3, 0.3],
            )
            mesh = pyrender.Mesh.from_trimesh(deformed, smooth=False)
            scene.add(mesh)

            # Lights
            scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0))

            # 3. For each camera, set pose, render
            for spec_cam, (cam_node, _intr) in zip(camera_specs, zip(cam_nodes_per_view, intr_per_view)):
                # Each cam_node holds its own pose; we add the camera as the active
                cam = pyrender.IntrinsicsCamera(
                    fx=_intr["fx"], fy=_intr["fy"],
                    cx=_intr["cx"], cy=_intr["cy"],
                    znear=0.05, zfar=20.0,
                )
                node = scene.add(cam, pose=cam_node["c2w_opengl"])
                rgb, _depth = renderer.render(scene)
                scene.remove_node(node)
                writers[spec_cam.name].append_data(rgb)

        for w in writers.values():
            w.close()
    finally:
        renderer.delete()

    # ---- write metadata
    cameras_meta = {}
    for cam_node, intr in zip(cam_nodes_per_view, intr_per_view):
        # Convert OpenGL camera-to-world to world-to-camera in OpenCV convention
        c2w_gl = cam_node["c2w_opengl"]
        # OpenGL <-> OpenCV: flip Y and Z of camera frame
        gl_to_cv = np.diag([1, -1, -1, 1])
        c2w_cv = c2w_gl @ gl_to_cv
        w2c_cv = np.linalg.inv(c2w_cv)
        cameras_meta[cam_node["name"]] = {
            "intrinsics": intr,
            "extrinsics": {"world_to_camera_4x4": w2c_cv.tolist()},
        }
    with open(out_dir / "cameras.json", "w") as f:
        json.dump(cameras_meta, f, indent=2)

    # trajectory.npz: just frame indices for soft tasks (no joint qpos)
    np.savez(
        out_dir / "trajectory.npz",
        frame_idx=np.arange(n_frames, dtype=np.int32),
    )

    with open(out_dir / "physics.json", "w") as f:
        json.dump(traj_record["physics_params"], f, indent=2)

    meta = {
        "traj_id": traj_record["traj_id"],
        "obj_id": traj_record["obj_id"],
        "obj_category": traj_record["obj_category"],
        "task_name": traj_record["task_name"],
        "n_frames": n_frames,
        "fps": fps,
        "image_size": image_size,
        "seed": traj_record["seed"],
        "success": traj_record["success"],
        "object_type": traj_record["object_type"],
        "soft_object_spec": traj_record["soft_object_spec"],
        "deformation_params_per_frame": deform_seq,
        "is_composition": traj_record.get("is_composition", False),
        "composition_steps": traj_record.get("composition_steps", []),
        "eval_only": traj_record.get("eval_only", False),
        "randomization": traj_record.get("randomization", {}),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return SoftRenderResult(
        traj_id=traj_record["traj_id"],
        out_dir=out_dir,
        n_frames_written=n_frames,
        cameras_meta=cameras_meta,
    )


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _apply_deformation(rest_mesh, task_name: str, deform: dict):
    """Dispatch on task name to apply the right deformation."""
    if task_name == "squeeze":
        return apply_anisotropic_scale(rest_mesh, deform.get("scale_xyz", [1, 1, 1]))
    if task_name == "fold":
        return apply_hinge_fold(
            rest_mesh,
            hinge_axis=deform.get("hinge_axis", [1, 0, 0]),
            hinge_origin=deform.get("hinge_origin", [0, 0, 0]),
            fold_angle_rad=float(deform.get("fold_angle_rad", 0.0)),
            side_to_fold=deform.get("side_to_fold", "positive"),
        )
    # Unknown task: identity
    return rest_mesh


def _estimate_diag(mesh) -> float:
    bb = mesh.bounds
    return float(np.linalg.norm(bb[1] - bb[0]))


def _build_camera_nodes(camera_specs: List[CameraSpec],
                         *,
                         image_size: int,
                         bbox_diag: float):
    """Translate CameraSpec list into (cam_node_info, intrinsics) for pyrender.

    Returns:
        cam_nodes_per_view: list of {"name": ..., "c2w_opengl": (4,4)}
        intr_per_view:      list of {"fx", "fy", "cx", "cy", "width", "height"}

    pyrender uses OpenGL convention: camera looks down -Z, +Y up.
    """
    from scipy.spatial.transform import Rotation as R

    nodes, intrs = [], []
    for spec in camera_specs:
        # Spherical coords relative to object origin
        az = np.deg2rad(spec.azimuth_deg)
        el = np.deg2rad(spec.elevation_deg)
        radius = max(bbox_diag * spec.distance_factor, 0.5)
        cam_pos = np.array([
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            radius * np.sin(el),
        ])

        # Look at origin
        target = np.array([0, 0, 0], dtype=np.float64)
        # OpenGL camera convention
        forward = (cam_pos - target)   # camera looks DOWN forward axis
        forward = forward / (np.linalg.norm(forward) + 1e-9)
        up_world = np.array([0, 0, 1.0])
        if abs(np.dot(forward, up_world)) > 0.999:
            up_world = np.array([1, 0, 0.0])
        right = np.cross(up_world, forward)
        right = right / (np.linalg.norm(right) + 1e-9)
        new_up = np.cross(forward, right)

        # OpenGL: x=right, y=up, z=back (away from scene)
        rot = np.eye(3)
        rot[:, 0] = right
        rot[:, 1] = new_up
        rot[:, 2] = forward    # backward

        c2w = np.eye(4)
        c2w[:3, :3] = rot
        c2w[:3, 3] = cam_pos

        # Intrinsics from FOV
        fovy = np.deg2rad(spec.fovy_deg)
        fy = image_size / (2 * np.tan(fovy / 2))
        intr = {
            "fx": float(fy), "fy": float(fy),
            "cx": float(image_size / 2), "cy": float(image_size / 2),
            "width": image_size, "height": image_size,
        }
        nodes.append({"name": spec.name, "c2w_opengl": c2w})
        intrs.append(intr)
    return nodes, intrs
