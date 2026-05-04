"""Motion-saliency scoring.

Filters out PartNet-Mobility instances whose articulation produces nearly
invisible pixel changes — those would give the encoder no learning signal and
trivially satisfy the algebraic losses, leading to codebook collapse.

Score has three components, multiplied together:
  1. geometry: how much the moving part sweeps through space (arc length or
                slide distance) relative to the object's bbox diagonal.
  2. volume:   ratio of moving-part volume to total object volume.
  3. pixel:    mean per-pixel RGB difference between joint=0 and joint=max,
                rendered from the configured cameras. Normalized to [0,1].

Set thresholds in configs/default.yaml under `saliency:`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .object_loader import PartNetObject, JointInfo

logger = logging.getLogger(__name__)


@dataclass
class SaliencyScore:
    obj_id: str
    category: str
    joint_index: int
    geometry_score: float
    volume_score: float
    pixel_score: float
    total: float

    def passes(self, cfg: dict) -> bool:
        return (
            self.geometry_score >= cfg["min_geometry_score"]
            and self.volume_score   >= cfg["min_volume_ratio"]
            and self.pixel_score    >= cfg["min_pixel_diff"]
        )


def geometry_score_for_joint(joint: "JointInfo", bbox_diag: float,
                              moving_part_radius: float = 0.3) -> float:
    """For revolute joints, score = arc_length / bbox_diag.
    For prismatic joints, score = displacement / bbox_diag.

    `moving_part_radius` is a rough estimate (in meters) of how far the moving
    part center is from the joint axis; we default to 0.3 m which is realistic
    for cabinet doors / drawers.
    """
    if bbox_diag <= 0:
        return 0.0
    if joint.joint_type in ("revolute", "continuous"):
        arc = abs(joint.range) * moving_part_radius
        return arc / bbox_diag
    elif joint.joint_type == "prismatic":
        return abs(joint.range) / bbox_diag
    return 0.0


def volume_score_for_joint(volume_ratio: float) -> float:
    """Direct passthrough; volume_ratio is moving_part_volume / total_volume."""
    return float(volume_ratio)


def pixel_diff_score(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Mean per-pixel L1 difference of two RGB images normalized to [0,1].

    Both images must be uint8 (H,W,3) or float (H,W,3) in [0,1].
    Returns a scalar in [0,1].
    """
    a = rgb_a.astype(np.float32)
    b = rgb_b.astype(np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    if b.max() > 1.5:
        b = b / 255.0
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.abs(a - b).mean())


# ----------------------------------------------------------------------------
# top-level: score one object
# ----------------------------------------------------------------------------
def score_object(
    obj: "PartNetObject",
    *,
    renderer,                 # callable: (obj, joint_idx, qpos) -> rgb (H,W,3)
    joint_index: Optional[int] = None,
) -> Optional[SaliencyScore]:
    """Compute the 3-component saliency score for one object.

    Args:
        obj: PartNetObject metadata.
        renderer: a callable that, given (obj, joint_index, qpos), returns
                  one rendered RGB image. Pass `make_quick_renderer()` from
                  this module to get a default SAPIEN-based one.
        joint_index: which active joint to test; defaults to the largest one.

    Returns:
        SaliencyScore, or None if the object has no usable joint.
    """
    if not obj.has_active_joint:
        return None

    joint = obj.largest_joint() if joint_index is None else obj.joints[joint_index]
    if joint is None or not joint.is_active:
        return None

    # 1. geometry component
    geom = geometry_score_for_joint(joint, obj.bbox_diagonal)

    # 2. volume component (rough estimate from heuristic; real value would need
    #    per-link mesh decomposition. We approximate as 1/N for now.)
    n_links = max(len(obj.joints) + 1, 2)
    vol = 1.0 / n_links

    # 3. pixel-diff component: render at qmin and at qmax
    try:
        rgb_min = renderer(obj, joint, joint.limit_low)
        rgb_max = renderer(obj, joint, joint.limit_high)
        pix = pixel_diff_score(rgb_min, rgb_max)
    except Exception as e:  # noqa: BLE001
        logger.warning("Render failed for %s: %s", obj.obj_id, e)
        return None

    total = geom * vol * pix * 100.0  # rescale for readability
    return SaliencyScore(
        obj_id=obj.obj_id,
        category=obj.category,
        joint_index=obj.joints.index(joint),
        geometry_score=geom,
        volume_score=vol,
        pixel_score=pix,
        total=total,
    )


# ----------------------------------------------------------------------------
# default renderer: minimal SAPIEN setup
# ----------------------------------------------------------------------------
def make_quick_renderer(image_size: int = 128, distance_factor: float = 2.5):
    """Return a callable (obj, joint, qpos) -> rgb suitable for saliency tests.

    Uses a single front-elevated camera at `image_size` resolution for speed.
    Spins up SAPIEN lazily on first call and reuses the engine across calls.
    """
    state = {"engine": None, "renderer": None}

    def _setup():
        import sapien.core as sapien
        engine = sapien.Engine()
        sap_renderer = sapien.SapienRenderer()
        engine.set_renderer(sap_renderer)
        state["engine"] = engine
        state["renderer"] = sap_renderer

    def _render(obj, joint, qpos: float) -> np.ndarray:
        if state["engine"] is None:
            _setup()
        import sapien.core as sapien

        engine = state["engine"]
        scene = engine.create_scene()
        scene.set_timestep(0.01)
        scene.add_ground(altitude=-1.0)
        scene.set_ambient_light([0.4, 0.4, 0.4])
        scene.add_directional_light([0, 0, -1], [0.6, 0.6, 0.6])

        articulation = obj.load(scene, fix_root=True)

        # Set qpos for the requested joint
        active_joints = articulation.get_active_joints()
        idx_in_active = None
        for i, j in enumerate(active_joints):
            if j.name == joint.name:
                idx_in_active = i
                break
        if idx_in_active is None:
            idx_in_active = 0  # fallback

        qpos_full = articulation.get_qpos()
        qpos_full[idx_in_active] = qpos
        articulation.set_qpos(qpos_full)

        # Camera (front-elevated)
        d = max(obj.bbox_diagonal * distance_factor, 1.0)
        cam = scene.add_camera(
            name="cam_saliency",
            width=image_size, height=image_size,
            fovy=np.deg2rad(45),
            near=0.05, far=100.0,
        )
        # Look-at: from (d, 0, d/2) toward origin
        cam.set_pose(sapien.Pose(p=[d, 0, d * 0.4]))
        # Point camera at origin (object root)
        _look_at(cam, target=[0, 0, 0], up=[0, 0, 1])

        scene.update_render()
        cam.take_picture()
        rgba = cam.get_color_rgba()  # (H, W, 4) float in [0,1]
        return rgba[..., :3]

    return _render


def _look_at(camera, target, up=(0, 0, 1)):
    """Aim a SAPIEN camera at a target point."""
    import sapien.core as sapien

    pos = np.array(camera.get_pose().p)
    tgt = np.array(target)
    forward = (tgt - pos)
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up = np.array(up, dtype=np.float64)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    new_up = np.cross(right, forward)

    # SAPIEN camera convention: x-forward, y-left, z-up (mat columns)
    rot = np.eye(3)
    rot[:, 0] = forward
    rot[:, 1] = -right
    rot[:, 2] = new_up
    from scipy.spatial.transform import Rotation as R  # type: ignore[import]
    quat = R.from_matrix(rot).as_quat()  # xyzw
    # SAPIEN uses wxyz order
    pose = sapien.Pose(p=pos, q=[quat[3], quat[0], quat[1], quat[2]])
    camera.set_pose(pose)
