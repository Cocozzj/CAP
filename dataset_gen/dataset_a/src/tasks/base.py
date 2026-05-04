"""Base class for all task templates.

A Task generates a *trajectory*: a fixed-length sequence of joint setpoints
(13 seconds @ 30 fps = 390 frames per the PDF spec). The frames are split into:

    [pre_settle]   object held in start state, no motion
    [motion]       min-jerk interpolation from qpos_start to qpos_end
    [post_settle]  object held in end state, no motion

Lengths of the three parts are sampled per trajectory from the task config:
    pre_settle_range          (defaults block)
    motion_duration_range     (per-task)
    post_settle_range         (defaults block, may be auto-shrunk to fit)

Total = total_duration_seconds × fps frames, always.

Trajectory generation is separated from rendering: this class only produces
joint-angle sequences and physics parameters. The renderer (Step 3) replays
those sequences through SAPIEN to produce RGB videos.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# trajectory record
# ----------------------------------------------------------------------------
@dataclass
class TrajectoryRecord:
    """One generated trajectory for a single (object × task)."""

    traj_id: str
    obj_id: str
    obj_category: str
    obj_folder: str
    task_name: str
    joint_index: int
    joint_name: str
    success: bool
    n_frames: int
    fps: int
    seed: int

    # Per-frame data (length == n_frames)
    joint_qpos: List[float] = field(default_factory=list)
    joint_qpos_actual: List[float] = field(default_factory=list)
    object_pose_world: List[List[float]] = field(default_factory=list)

    # Physics params (constant across the trajectory)
    physics_params: dict = field(default_factory=dict)

    # Phase boundaries (for downstream subgoal labeling)
    pre_settle_frames: int = 0
    motion_frames: int = 0
    post_settle_frames: int = 0

    # Composition fields. For atomic trajectories these are defaults.
    is_composition: bool = False
    composition_steps: List[str] = field(default_factory=list)   # e.g. ["open", "close"]
    sub_action_frame_ranges: List[List[int]] = field(default_factory=list)  # [(start, end), ...]
    eval_only: bool = False         # if True, splitter routes to test_compositional_long

    # Soft body fields. For articulated trajectories these are defaults.
    object_type: str = "articulated"   # "articulated" | "soft"
    soft_object_spec: dict = field(default_factory=dict)   # see soft_objects.SoftObjectSpec.to_dict
    deformation_params_per_frame: List[dict] = field(default_factory=list)
    # Used when soft body is attached to an articulated host (e.g. Kettle pour
    # uses an articulated kettle + tilt root pose). Default empty.

    # Task-specific randomization that produced this trajectory
    randomization: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "obj_id": self.obj_id,
            "obj_category": self.obj_category,
            "obj_folder": self.obj_folder,
            "task_name": self.task_name,
            "joint_index": self.joint_index,
            "joint_name": self.joint_name,
            "success": self.success,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "seed": self.seed,
            "joint_qpos": self.joint_qpos,
            "joint_qpos_actual": self.joint_qpos_actual,
            "object_pose_world": self.object_pose_world,
            "physics_params": self.physics_params,
            "pre_settle_frames": self.pre_settle_frames,
            "motion_frames": self.motion_frames,
            "post_settle_frames": self.post_settle_frames,
            "is_composition": self.is_composition,
            "composition_steps": self.composition_steps,
            "sub_action_frame_ranges": self.sub_action_frame_ranges,
            "eval_only": self.eval_only,
            "object_type": self.object_type,
            "soft_object_spec": self.soft_object_spec,
            "deformation_params_per_frame": self.deformation_params_per_frame,
            "randomization": self.randomization,
        }


# ----------------------------------------------------------------------------
# velocity profiles
# ----------------------------------------------------------------------------
def minimum_jerk_profile(t: np.ndarray, T: float) -> np.ndarray:
    """Smooth 0->1 trajectory in [0, T]. Standard min-jerk."""
    s = np.clip(t / T, 0.0, 1.0)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


# ----------------------------------------------------------------------------
# base class
# ----------------------------------------------------------------------------
class BaseTask:
    """Subclasses implement `compute_target_qpos()`."""

    NAME: str = "base"

    def __init__(self, task_cfg: dict, defaults_cfg: dict, physics_cfg: dict):
        """
        Args:
            task_cfg:     one entry of `tasks.yaml#tasks`
            defaults_cfg: the `defaults` block of `tasks.yaml`
            physics_cfg:  the `physics` block of `default.yaml`
        """
        self.cfg = task_cfg
        self.defaults = defaults_cfg
        self.physics_cfg = physics_cfg
        if task_cfg.get("name", self.NAME) != self.NAME:
            logger.warning("task_cfg name '%s' != %s",
                           task_cfg.get("name"), self.NAME)

    # --------------------------------------------------------- subclass API
    def compute_target_qpos(
        self,
        joint_low: float,
        joint_high: float,
        rng: random.Random,
    ) -> Tuple[float, float, dict]:
        """Return (qpos_start, qpos_end, randomization_dict)."""
        raise NotImplementedError

    # --------------------------------------------------------- main entry
    def generate(
        self,
        obj_record: dict,
        seed: int,
        traj_id: str,
        scene,
    ) -> Optional[TrajectoryRecord]:
        """Run the task on a SAPIEN scene; returns a TrajectoryRecord or None."""
        rng = random.Random(seed)

        joint_low = obj_record["joint_limit_low"]
        joint_high = obj_record["joint_limit_high"]

        try:
            qpos_start, qpos_end, rand_info = self.compute_target_qpos(
                joint_low, joint_high, rng
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("compute_target_qpos failed: %s", e)
            return None

        # Load articulation
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = str(obj_record["folder"]) + "/mobility.urdf"
        articulation = loader.load(urdf_path)
        if articulation is None:
            logger.warning("Failed to load %s", obj_record["folder"])
            return None

        # Find the active joint by name
        active_joints = articulation.get_active_joints()
        joint_idx_in_active = self._find_joint_index(
            active_joints, obj_record["joint_name"]
        )
        if joint_idx_in_active is None:
            logger.warning("No active joints in %s", obj_record["obj_id"])
            return None

        # Configure the drive
        target_joint = active_joints[joint_idx_in_active]
        stiffness = 1000.0
        damping = self.physics_cfg.get("default_damping", 0.1) * 100.0
        target_joint.set_drive_property(stiffness=stiffness, damping=damping)

        # Friction / damping randomization
        physics_params = self._sample_physics(rng)
        for j in active_joints:
            try:
                j.set_friction(physics_params["friction"])
            except Exception:
                pass

        # Sample trajectory phase durations
        fps = self.physics_cfg["fps"]
        total_seconds = self.defaults["total_duration_seconds"]
        n_frames_total = int(round(total_seconds * fps))

        pre_settle_s = rng.uniform(*self.defaults["pre_settle_range"])
        motion_s = rng.uniform(*self.cfg["motion_duration_range"])
        # speed factor scales motion (faster speed = shorter motion phase)
        speed_factor = rng.uniform(
            *self.cfg.get("randomize", {}).get("speed_factor_range", [1.0, 1.0])
        )
        motion_s = motion_s / speed_factor

        post_settle_s = total_seconds - pre_settle_s - motion_s
        if post_settle_s < self.defaults["post_settle_range"][0]:
            # Motion + pre too long; shrink motion to leave at least min post settle
            post_settle_s = self.defaults["post_settle_range"][0]
            motion_s = total_seconds - pre_settle_s - post_settle_s
            if motion_s < 0.5:
                # Couldn't fit; shrink pre too
                pre_settle_s = max(0.5, total_seconds - 0.5 - post_settle_s)
                motion_s = total_seconds - pre_settle_s - post_settle_s

        n_pre = int(round(pre_settle_s * fps))
        n_motion = int(round(motion_s * fps))
        n_post = n_frames_total - n_pre - n_motion
        if n_post < 0:
            n_motion += n_post  # absorb into motion if any rounding spillover
            n_post = 0

        # Build qpos sequence: [start]*n_pre + min-jerk + [end]*n_post
        qpos_seq = np.empty(n_frames_total, dtype=np.float64)
        qpos_seq[:n_pre] = qpos_start
        if n_motion > 0:
            t = np.arange(n_motion) / fps
            s = minimum_jerk_profile(t, T=max(motion_s, 1e-3))
            qpos_seq[n_pre : n_pre + n_motion] = qpos_start + s * (qpos_end - qpos_start)
        qpos_seq[n_pre + n_motion :] = qpos_end

        # Set initial qpos
        qpos_full = np.zeros(len(active_joints), dtype=np.float64)
        qpos_full[joint_idx_in_active] = qpos_start
        articulation.set_qpos(qpos_full)

        # Step physics, recording achieved qpos and root pose
        steps_per_frame = self.physics_cfg.get("steps_per_frame", 10)
        scene.set_timestep(self.physics_cfg.get("step_dt", 0.0033))

        joint_qpos_actual: List[float] = []
        object_pose_world: List[List[float]] = []

        for q_target in qpos_seq:
            target_joint.set_drive_target(float(q_target))
            for _ in range(steps_per_frame):
                scene.step()
            achieved = float(articulation.get_qpos()[joint_idx_in_active])
            joint_qpos_actual.append(achieved)
            root_pose = articulation.get_root_pose()
            object_pose_world.append([
                root_pose.p[0], root_pose.p[1], root_pose.p[2],
                root_pose.q[0], root_pose.q[1], root_pose.q[2], root_pose.q[3],
            ])

        # Success criterion: at the FINAL frame (after post_settle, the joint
        # has fully settled), the achieved qpos has covered at least 50% of
        # the targeted motion, in signed terms. We deliberately do NOT use
        # the end-of-motion frame, because high-inertia objects (fridge doors,
        # drawers) lag the min-jerk reference and only converge during the
        # post_settle phase. We use signed progress (achieved-start)/(end-start)
        # rather than absolute error so that PartNet URDFs which store joint
        # limits with low > high (sign-flipped) are handled the same as normal.
        if n_frames_total > 0 and len(joint_qpos_actual) > 0:
            achieved_final = joint_qpos_actual[-1]
            total_motion = qpos_end - qpos_start
            if abs(total_motion) < 1e-6:
                success = True  # degenerate task
            else:
                signed_progress = (achieved_final - qpos_start) / total_motion
                success = signed_progress >= 0.5
        else:
            success = False

        return TrajectoryRecord(
            traj_id=traj_id,
            obj_id=obj_record["obj_id"],
            obj_category=obj_record["our_category"],
            obj_folder=str(obj_record["folder"]),
            task_name=self.NAME,
            joint_index=obj_record["joint_index"],
            joint_name=obj_record["joint_name"],
            success=success,
            n_frames=n_frames_total,
            fps=fps,
            seed=seed,
            joint_qpos=qpos_seq.tolist(),
            joint_qpos_actual=joint_qpos_actual,
            object_pose_world=object_pose_world,
            physics_params=physics_params,
            pre_settle_frames=n_pre,
            motion_frames=n_motion,
            post_settle_frames=n_post,
            randomization={
                **rand_info,
                "pre_settle_seconds": pre_settle_s,
                "motion_seconds": motion_s,
                "post_settle_seconds": post_settle_s,
                "speed_factor": speed_factor,
            },
        )

    # --------------------------------------------------------- helpers
    @staticmethod
    def _find_joint_index(active_joints, joint_name: str) -> Optional[int]:
        if not active_joints:
            return None
        for i, j in enumerate(active_joints):
            if j.name == joint_name:
                return i
        # Fallback: pick the joint with the largest range
        ranges = []
        for i, j in enumerate(active_joints):
            try:
                lim = j.get_limits()
                if lim is not None and len(lim) > 0:
                    ranges.append((i, lim[0][1] - lim[0][0]))
            except Exception:
                continue
        if ranges:
            return max(ranges, key=lambda x: x[1])[0]
        return 0

    def _sample_physics(self, rng: random.Random) -> dict:
        cfg = self.physics_cfg
        out = {}
        f_lo, f_hi = cfg.get("randomize_friction",
                             [cfg.get("default_friction", 0.5)] * 2)
        out["friction"] = rng.uniform(f_lo, f_hi)
        d_lo, d_hi = cfg.get("randomize_damping",
                             [cfg.get("default_damping", 0.1)] * 2)
        out["damping"] = rng.uniform(d_lo, d_hi)
        if cfg.get("randomize_object_mass"):
            out["mass_scale"] = rng.uniform(*cfg["randomize_object_mass"])
        return out
