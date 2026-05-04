"""Multi-view synchronized renderer.

Given a saved trajectory (joint qpos sequence) and an object spec, replay it
in SAPIEN and capture RGB (and optionally depth) from N cameras, writing
H.264 MP4s.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .camera_setup import (
    CameraSpec,
    add_cameras_to_scene,
    camera_extrinsics,
    get_camera_intrinsics,
)
from .camera_planner import plan_cameras_for_articulation

logger = logging.getLogger(__name__)


# Per-process cache: {obj_id: (best_yaw_deg, ObjectAABB)} so we don't re-render
# the 4×2 motion-saliency probe for every trajectory of the same object.
_BEST_YAW_CACHE: dict = {}


@dataclass
class RenderResult:
    traj_id: str
    out_dir: Path
    n_frames_written: int
    cameras_meta: dict


# ----------------------------------------------------------------------------
# core renderer
# ----------------------------------------------------------------------------
def render_trajectory(
    traj_record: dict,
    obj_record: dict,
    camera_specs: List[CameraSpec],
    *,
    image_size: int = 256,
    save_depth: bool = False,
    out_dir: str | Path,
    fps: int = 30,
    camera_design: Optional[dict] = None,
) -> Optional[RenderResult]:
    """Render one trajectory to MP4 files. Dispatches to the soft renderer
    for soft trajectories (object_type == 'soft')."""
    # Dispatch on object_type
    if traj_record.get("object_type") == "soft":
        from .soft_renderer import render_soft_trajectory
        res = render_soft_trajectory(
            traj_record=traj_record,
            camera_specs=camera_specs,
            image_size=image_size,
            out_dir=out_dir,
            fps=fps,
        )
        if res is None:
            return None
        return RenderResult(
            traj_id=res.traj_id,
            out_dir=res.out_dir,
            n_frames_written=res.n_frames_written,
            cameras_meta=res.cameras_meta,
        )

    import sapien.core as sapien
    import imageio

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    engine = sapien.Engine()
    sap_renderer = sapien.SapienRenderer()
    engine.set_renderer(sap_renderer)

    scene = engine.create_scene()
    scene.set_timestep(1.0 / 300.0)
    # scene.add_ground(altitude=-1.0)   # removed: clean black background for init_gs
    scene.set_ambient_light([0.7, 0.7, 0.7])
    scene.add_directional_light([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])
    scene.add_directional_light([0.5, 0.5, -0.5], [0.3, 0.3, 0.3])

    # White backdrop: 6 faces enclosing scene (full white room).
    # Camera always sees white instead of SAPIEN's default black background.
    BACKDROP_HALF = 15.0    # half-size of the room (m)
    BACKDROP_THICK = 0.05
    BACKDROP_COLOR = [0.98, 0.98, 0.98, 1.0]
    for axis_idx, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        builder = scene.create_actor_builder()
        half = [BACKDROP_HALF, BACKDROP_HALF, BACKDROP_HALF]
        half[axis_idx] = BACKDROP_THICK
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=half,
            color=BACKDROP_COLOR,
        )
        actor = builder.build_static(name=f"backdrop_{axis_idx}_{sign}")
        pos = [0, 0, 0]
        pos[axis_idx] = sign * (BACKDROP_HALF - BACKDROP_THICK)
        actor.set_pose(sapien.Pose(pos))

    # Load articulation
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(Path(obj_record["folder"]) / "mobility.urdf"))
    if articulation is None:
        logger.error("Failed to load %s", obj_record["folder"])
        return None

    # Find the active joint
    active_joints = articulation.get_active_joints()
    target_idx = None
    for i, j in enumerate(active_joints):
        if j.name == obj_record["joint_name"]:
            target_idx = i
            break
    if target_idx is None and active_joints:
        target_idx = 0

    # Set joint to qpos_start so AABB / moving-part center reflect first frame
    qpos_full = np.zeros(len(active_joints), dtype=np.float64)
    if target_idx is not None and traj_record["joint_qpos"]:
        qpos_full[target_idx] = float(traj_record["joint_qpos"][0])
    articulation.set_qpos(qpos_full)

    # ---- cameras: adaptive per-object planner (preferred) or static fallback
    use_planner = bool(camera_design and camera_design.get("enable", False))
    if use_planner and target_idx is not None and traj_record["joint_qpos"]:
        # Look up cached (best_yaw, aabb) for this object_id
        obj_id = traj_record["obj_id"]
        qpos_low = float(min(traj_record["joint_qpos"]))
        qpos_high = float(max(traj_record["joint_qpos"]))
        if obj_id not in _BEST_YAW_CACHE:
            specs, target_xyz, aabb = plan_cameras_for_articulation(
                scene, articulation,
                target_joint_idx=target_idx,
                qpos_low=qpos_low, qpos_high=qpos_high,
                cfg=camera_design,
                seed=traj_record["seed"],
            )
            # Cache best_yaw + aabb so subsequent trajectories of the same
            # obj_id skip the 8-render motion probe
            from .camera_planner import compute_object_aabb
            # We re-derive best_yaw from the first specs (cam0 = best_yaw + offset)
            # To stably cache, recompute AABB only and re-find best_yaw the
            # first time, then for subsequent calls just plan with cached best_yaw.
            _BEST_YAW_CACHE[obj_id] = ({
                "specs_yaws_offset_baseline": [s.azimuth_deg for s in specs],
            }, aabb)
            # Restore qpos_start before rendering (probe call mutated it)
            qpos_full[target_idx] = float(traj_record["joint_qpos"][0])
            articulation.set_qpos(qpos_full)
        else:
            # We have cached AABB; re-plan with new jitter
            from .camera_planner import plan_cameras
            import random as _r
            cached_info, aabb = _BEST_YAW_CACHE[obj_id]
            # Use the first cached cam0 azimuth as a proxy for best_yaw
            best_yaw_proxy = cached_info["specs_yaws_offset_baseline"][0]
            rng = _r.Random(traj_record["seed"])
            specs, target_xyz = plan_cameras(aabb, best_yaw_proxy, rng, camera_design)

        # Now place cameras using the (possibly jittered) specs
        bbox_diag = aabb.diagonal
        cameras = add_cameras_to_scene(
            scene, specs,
            object_center=target_xyz,
            moving_part_center=target_xyz,   # both = AABB center
            bbox_diagonal=bbox_diag,
            image_size=image_size,
        )
    else:
        # Legacy static cameras path
        object_center = np.array(articulation.get_root_pose().p, dtype=np.float64)
        bbox_diag = obj_record.get("bbox_diagonal", 1.0)
        if bbox_diag < 1.5:
            bbox_diag = 2.0
        moving_center = _estimate_moving_part_center(articulation, target_idx)
        cameras = add_cameras_to_scene(
            scene, camera_specs,
            object_center=object_center,
            moving_part_center=moving_center,
            bbox_diagonal=bbox_diag,
            image_size=image_size,
        )
    # ---- prepare video writers
    writers = {}
    for name, _ in cameras:
        path = out_dir / f"{name}.mp4"
        writers[name] = imageio.get_writer(
            str(path), fps=fps, codec="libx264",
            macro_block_size=1, quality=8,
        )
    depth_buffers = {name: [] for name, _ in cameras} if save_depth else None

    # ---- replay trajectory
    is_root_pose_anim = (traj_record.get("object_type") == "articulated_root_pose")
    qpos_seq = traj_record["joint_qpos"]
    pose_seq = traj_record.get("object_pose_world", [])
    n_frames = max(len(qpos_seq), len(pose_seq))

    for f in range(n_frames):
        if is_root_pose_anim:
            # Drive the root pose (e.g. PourTask tilts the whole kettle)
            if f < len(pose_seq):
                p = pose_seq[f]
                # pose: [x, y, z, qw, qx, qy, qz]
                articulation.set_root_pose(
                    sapien.Pose(p=p[:3], q=p[3:7])
                )
        else:
            # Standard joint-qpos replay
            if f < len(qpos_seq):
                qpos_full[target_idx] = float(qpos_seq[f])
                articulation.set_qpos(qpos_full)

        scene.update_render()
        for name, cam in cameras:
            cam.take_picture()
            rgba = cam.get_color_rgba()
            rgb_uint8 = (rgba[..., :3] * 255).astype(np.uint8)
            writers[name].append_data(rgb_uint8)
            if save_depth:
                depth = cam.get_depth_image()
                depth_buffers[name].append(depth.astype(np.float16))

    for name in writers:
        writers[name].close()

    # ---- depth + per-frame metadata
    if save_depth:
        import h5py
        with h5py.File(out_dir / "depth.h5", "w") as f:
            for name in depth_buffers:
                stack = np.stack(depth_buffers[name], axis=0)
                f.create_dataset(name, data=stack, compression="lzf")

    # ---- camera metadata
    cameras_meta = {}
    for name, cam in cameras:
        cameras_meta[name] = {
            "intrinsics": get_camera_intrinsics(cam, image_size),
            "extrinsics": camera_extrinsics(cam),
        }
    with open(out_dir / "cameras.json", "w") as f:
        json.dump(cameras_meta, f, indent=2)

    # ---- per-frame trajectory data
    np.savez(
        out_dir / "trajectory.npz",
        joint_qpos=np.array(traj_record["joint_qpos"], dtype=np.float32),
        joint_qpos_actual=np.array(traj_record["joint_qpos_actual"], dtype=np.float32),
        object_pose_world=np.array(traj_record["object_pose_world"], dtype=np.float32),
    )

    # ---- physics + meta
    with open(out_dir / "physics.json", "w") as f:
        json.dump(traj_record["physics_params"], f, indent=2)

    meta = {
        "traj_id": traj_record["traj_id"],
        "obj_id": traj_record["obj_id"],
        "obj_folder": str(obj_record.get("folder", "")),
        "obj_category": traj_record["obj_category"],
        "task_name": traj_record["task_name"],
        "joint_index": traj_record["joint_index"],
        "joint_name": traj_record["joint_name"],
        "n_frames": traj_record["n_frames"],
        "fps": fps,
        "image_size": image_size,
        "seed": traj_record["seed"],
        "success": traj_record["success"],
        "randomization": traj_record["randomization"],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return RenderResult(
        traj_id=traj_record["traj_id"],
        out_dir=out_dir,
        n_frames_written=len(qpos_seq),
        cameras_meta=cameras_meta,
    )


def _estimate_moving_part_center(articulation, joint_idx: Optional[int]) -> np.ndarray:
    """Best-effort: return the world-pos of the link driven by the given joint."""
    if joint_idx is None:
        return np.array(articulation.get_root_pose().p, dtype=np.float64)
    try:
        active_joints = articulation.get_active_joints()
        joint = active_joints[joint_idx]
        child_link = joint.get_child_link()
        return np.array(child_link.get_pose().p, dtype=np.float64)
    except Exception:  # noqa: BLE001
        return np.array(articulation.get_root_pose().p, dtype=np.float64)
