"""Soft-body tasks: squeeze, fold, pour.

Implementation note (route A from earlier discussion):
This is *procedural* deformation, not real PBD/MPM physics. The trajectory
records per-frame deformation parameters (scale matrices, hinge angles, root
poses); the renderer applies them to the rest mesh and rasterizes the result.

This is enough for the model to learn that
    "squeeze token" ↔ object compresses
    "fold token"    ↔ cloth bends in half
    "pour token"    ↔ kettle tilts toward horizontal

For a paper that wants real differentiable physics for these (e.g. for the
physics-plugin section), upgrade to Warp / DiffTaichi later -- the trajectory
record shape doesn't change.

Tasks:
    SqueezeTask  -- compress a SoftToy / Sponge along an anisotropic axis
    FoldTask     -- bend a Cloth along a hinge line
    PourTask     -- tilt a Kettle (rigid root-pose animation) by 60-90 degrees
"""

from __future__ import annotations

import logging
import math
import random
from typing import List, Optional, Tuple

import numpy as np

from .base import (
    BaseTask,
    TrajectoryRecord,
    minimum_jerk_profile,
)

logger = logging.getLogger(__name__)


# ============================================================================
# common base for soft tasks
# ============================================================================
class SoftBaseTask:
    """Abstract base. Subclasses implement `compute_deformation_sequence`."""

    NAME: str = "soft_base"

    def __init__(self, task_cfg: dict, defaults_cfg: dict, physics_cfg: dict):
        self.cfg = task_cfg
        self.defaults = defaults_cfg
        self.physics_cfg = physics_cfg

    # --------------------------------------------------------- subclass API
    def compute_deformation_sequence(
        self,
        obj_record: dict,
        rng: random.Random,
        n_frames_total: int,
        n_pre: int,
        n_motion: int,
        n_post: int,
    ) -> Tuple[List[dict], dict]:
        """Return (per_frame_params_list, randomization_info_dict)."""
        raise NotImplementedError

    # --------------------------------------------------------- main entry
    def generate(
        self,
        obj_record: dict,
        seed: int,
        traj_id: str,
        scene,                  # unused for procedural soft tasks
    ) -> Optional[TrajectoryRecord]:
        rng = random.Random(seed)
        fps = self.physics_cfg["fps"]

        # Time partition: same scheme as articulated BaseTask
        total_seconds = self.defaults["total_duration_seconds"]
        n_frames_total = int(round(total_seconds * fps))

        pre_settle_s = rng.uniform(*self.defaults["pre_settle_range"])
        motion_s = rng.uniform(*self.cfg["motion_duration_range"])
        speed_factor = rng.uniform(
            *self.cfg.get("randomize", {}).get("speed_factor_range", [1.0, 1.0])
        )
        motion_s = motion_s / speed_factor

        post_settle_s = total_seconds - pre_settle_s - motion_s
        if post_settle_s < self.defaults["post_settle_range"][0]:
            post_settle_s = self.defaults["post_settle_range"][0]
            motion_s = max(0.5, total_seconds - pre_settle_s - post_settle_s)

        n_pre = int(round(pre_settle_s * fps))
        n_motion = int(round(motion_s * fps))
        n_post = max(0, n_frames_total - n_pre - n_motion)

        try:
            deform_seq, rand_info = self.compute_deformation_sequence(
                obj_record, rng, n_frames_total, n_pre, n_motion, n_post,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("compute_deformation_sequence failed: %s", e)
            return None

        if len(deform_seq) != n_frames_total:
            # Pad / truncate to exact length
            if len(deform_seq) < n_frames_total:
                last = deform_seq[-1] if deform_seq else {}
                deform_seq = deform_seq + [last] * (n_frames_total - len(deform_seq))
            else:
                deform_seq = deform_seq[:n_frames_total]

        physics_params = self._sample_physics(rng)

        return TrajectoryRecord(
            traj_id=traj_id,
            obj_id=obj_record["obj_id"],
            obj_category=obj_record["our_category"],
            obj_folder=str(obj_record.get("folder", "")),
            task_name=self.NAME,
            joint_index=obj_record.get("joint_index", -1),
            joint_name=obj_record.get("joint_name", ""),
            success=True,           # procedural deformation always "succeeds"
            n_frames=n_frames_total,
            fps=fps,
            seed=seed,
            joint_qpos=[],          # not applicable
            joint_qpos_actual=[],
            object_pose_world=[],
            physics_params=physics_params,
            pre_settle_frames=n_pre,
            motion_frames=n_motion,
            post_settle_frames=n_post,
            object_type="soft",
            soft_object_spec=obj_record.get("soft_object_spec", {}),
            deformation_params_per_frame=deform_seq,
            randomization={
                **rand_info,
                "pre_settle_seconds": pre_settle_s,
                "motion_seconds": motion_s,
                "post_settle_seconds": post_settle_s,
                "speed_factor": speed_factor,
            },
        )

    # ---------------------------------------------------------
    def _sample_physics(self, rng: random.Random) -> dict:
        cfg = self.physics_cfg
        out = {}
        f_lo, f_hi = cfg.get("randomize_friction",
                             [cfg.get("default_friction", 0.5)] * 2)
        out["friction"] = rng.uniform(f_lo, f_hi)
        d_lo, d_hi = cfg.get("randomize_damping",
                             [cfg.get("default_damping", 0.1)] * 2)
        out["damping"] = rng.uniform(d_lo, d_hi)
        return out


# ============================================================================
# Squeeze: anisotropic compression along a random axis
# ============================================================================
class SqueezeTask(SoftBaseTask):
    NAME = "squeeze"

    def compute_deformation_sequence(self, obj_record, rng, n_total, n_pre, n_motion, n_post):
        """Per-frame: dict with 'scale_xyz' = [sx, sy, sz]."""
        target_compression_range = self.cfg.get("target_compression_range", [0.4, 0.7])
        compress_to = rng.uniform(*target_compression_range)
        squeeze_axis = rng.choice(["x", "y", "z"])

        # Build scale interpolation
        rest_scale = np.array([1.0, 1.0, 1.0])
        target_scale = np.array([1.0, 1.0, 1.0])
        axis_idx = "xyz".index(squeeze_axis)
        target_scale[axis_idx] = compress_to
        # When compressing one axis, slightly bulge the others (volume-preserving feel)
        bulge_factor = 1.0 / math.sqrt(compress_to)
        for k in range(3):
            if k != axis_idx:
                target_scale[k] = bulge_factor

        deform_seq = []
        # pre: rest
        for _ in range(n_pre):
            deform_seq.append({"scale_xyz": rest_scale.tolist()})

        # motion: min-jerk between rest and target
        if n_motion > 0:
            t = np.arange(n_motion) / max(self.physics_cfg["fps"], 1)
            T = max(n_motion / self.physics_cfg["fps"], 1e-3)
            s = minimum_jerk_profile(t, T=T)
            for s_val in s:
                cur = rest_scale + s_val * (target_scale - rest_scale)
                deform_seq.append({"scale_xyz": cur.tolist()})

        # post: hold target
        for _ in range(n_post):
            deform_seq.append({"scale_xyz": target_scale.tolist()})

        rand_info = {
            "squeeze_axis": squeeze_axis,
            "target_compression": compress_to,
            "bulge_factor": bulge_factor,
        }
        return deform_seq, rand_info


# ============================================================================
# Fold: hinge bend on a cloth along its X axis
# ============================================================================
class FoldTask(SoftBaseTask):
    NAME = "fold"

    def compute_deformation_sequence(self, obj_record, rng, n_total, n_pre, n_motion, n_post):
        """Per-frame: dict with 'fold_angle_rad', 'hinge_axis', 'hinge_origin', 'side_to_fold'."""
        fold_deg_range = self.cfg.get("fold_angle_range", [120, 180])
        fold_deg = rng.uniform(*fold_deg_range)
        fold_target_rad = math.radians(fold_deg)

        # For our cloth (lies in XZ plane after rotation), hinge along X axis
        hinge_axis = [1.0, 0.0, 0.0]
        hinge_origin = [0.0, 0.0, 0.0]
        side_to_fold = rng.choice(["positive", "negative"])

        deform_seq = []
        for _ in range(n_pre):
            deform_seq.append({
                "fold_angle_rad": 0.0,
                "hinge_axis": hinge_axis,
                "hinge_origin": hinge_origin,
                "side_to_fold": side_to_fold,
            })

        if n_motion > 0:
            t = np.arange(n_motion) / max(self.physics_cfg["fps"], 1)
            T = max(n_motion / self.physics_cfg["fps"], 1e-3)
            s = minimum_jerk_profile(t, T=T)
            sign = 1.0 if side_to_fold == "positive" else -1.0
            for s_val in s:
                deform_seq.append({
                    "fold_angle_rad": float(sign * s_val * fold_target_rad),
                    "hinge_axis": hinge_axis,
                    "hinge_origin": hinge_origin,
                    "side_to_fold": side_to_fold,
                })

        sign = 1.0 if side_to_fold == "positive" else -1.0
        for _ in range(n_post):
            deform_seq.append({
                "fold_angle_rad": float(sign * fold_target_rad),
                "hinge_axis": hinge_axis,
                "hinge_origin": hinge_origin,
                "side_to_fold": side_to_fold,
            })

        rand_info = {
            "fold_angle_deg": fold_deg,
            "side_to_fold": side_to_fold,
        }
        return deform_seq, rand_info


# ============================================================================
# Pour: rigid root-pose tilt of a kettle (uses articulated SAPIEN object,
# but we record it as soft-style root-pose animation since the joint isn't
# being driven).
# ============================================================================
class PourTask(BaseTask):
    """Tilt the whole kettle by rotating its root pose.

    Inherits from BaseTask (not SoftBaseTask) because kettle is an articulated
    PartNet object. We override generate() to drive the *root* pose instead of
    a joint, which is what 'pour' physically means (whole object rotates).
    """

    NAME = "pour"

    def compute_target_qpos(self, joint_low, joint_high, rng):
        # Not used (we override generate)
        return 0.0, 0.0, {"direction": "pour_root_tilt"}

    def generate(self, obj_record, seed, traj_id, scene):
        rng = random.Random(seed)
        fps = self.physics_cfg["fps"]

        # Time partition (same scheme as BaseTask)
        total_seconds = self.defaults["total_duration_seconds"]
        n_frames_total = int(round(total_seconds * fps))
        pre_settle_s = rng.uniform(*self.defaults["pre_settle_range"])
        motion_s = rng.uniform(*self.cfg["motion_duration_range"])
        speed_factor = rng.uniform(
            *self.cfg.get("randomize", {}).get("speed_factor_range", [1.0, 1.0])
        )
        motion_s = motion_s / speed_factor
        post_settle_s = total_seconds - pre_settle_s - motion_s
        if post_settle_s < self.defaults["post_settle_range"][0]:
            post_settle_s = self.defaults["post_settle_range"][0]
            motion_s = max(0.5, total_seconds - pre_settle_s - post_settle_s)

        n_pre = int(round(pre_settle_s * fps))
        n_motion = int(round(motion_s * fps))
        n_post = max(0, n_frames_total - n_pre - n_motion)

        # Tilt parameters
        tilt_deg_range = self.cfg.get("tilt_angle_range", [60, 90])
        tilt_target_deg = rng.uniform(*tilt_deg_range)
        tilt_target_rad = math.radians(tilt_target_deg)
        tilt_axis = rng.choice(["x", "y"])   # tip forward or sideways

        # Build per-frame root pose (we encode the root pose as a separate
        # field in object_pose_world; renderer uses it instead of joint qpos)
        from scipy.spatial.transform import Rotation as R

        object_pose_world = []
        # Initial root pose (assume identity at origin; adjust per-render)
        for _ in range(n_pre):
            quat = R.from_rotvec([0, 0, 0]).as_quat()  # identity
            # SAPIEN uses wxyz
            object_pose_world.append([0, 0, 0, 1, 0, 0, 0])

        if n_motion > 0:
            t = np.arange(n_motion) / fps
            T = max(motion_s, 1e-3)
            s = minimum_jerk_profile(t, T=T)
            for s_val in s:
                ang = s_val * tilt_target_rad
                if tilt_axis == "x":
                    rotvec = [ang, 0, 0]
                else:
                    rotvec = [0, ang, 0]
                q_xyzw = R.from_rotvec(rotvec).as_quat()
                # wxyz
                object_pose_world.append([0, 0, 0, q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # Hold final tilted pose
        if tilt_axis == "x":
            rotvec_final = [tilt_target_rad, 0, 0]
        else:
            rotvec_final = [0, tilt_target_rad, 0]
        q_final_xyzw = R.from_rotvec(rotvec_final).as_quat()
        for _ in range(n_post):
            object_pose_world.append(
                [0, 0, 0, q_final_xyzw[3], q_final_xyzw[0], q_final_xyzw[1], q_final_xyzw[2]]
            )

        physics_params = self._sample_physics(rng)

        return TrajectoryRecord(
            traj_id=traj_id,
            obj_id=obj_record["obj_id"],
            obj_category=obj_record["our_category"],
            obj_folder=str(obj_record["folder"]),
            task_name=self.NAME,
            joint_index=obj_record.get("joint_index", -1),
            joint_name=obj_record.get("joint_name", ""),
            success=True,
            n_frames=n_frames_total,
            fps=fps,
            seed=seed,
            joint_qpos=[],            # joint not driven
            joint_qpos_actual=[],
            object_pose_world=object_pose_world,
            physics_params=physics_params,
            pre_settle_frames=n_pre,
            motion_frames=n_motion,
            post_settle_frames=n_post,
            object_type="articulated_root_pose",   # special: tells renderer to drive root pose
            randomization={
                "tilt_axis": tilt_axis,
                "tilt_target_deg": tilt_target_deg,
                "pre_settle_seconds": pre_settle_s,
                "motion_seconds": motion_s,
                "post_settle_seconds": post_settle_s,
            },
        )
