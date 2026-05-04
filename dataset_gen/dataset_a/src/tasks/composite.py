"""Composite task: chain N atomic actions on the same joint of the same object.

Used for both:
  - 2-step compositions (in train + eval; PDF "组合泛化曲线" 2-step point)
  - 3-step+ compositions marked eval_only (zero-shot generalization to longer chains)

Time budget for a composition with N steps:

    [pre_settle] [motion_1] [inter] [motion_2] [inter] ... [motion_N] [post_settle]

Total = total_duration_seconds (always 13s per PDF).
Motion budget is split equally across N steps; pre/inter/post are sampled.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseTask, TrajectoryRecord, minimum_jerk_profile

logger = logging.getLogger(__name__)


class CompositeTask:
    """Sequence of atomic tasks. Not a subclass of BaseTask because the
    generation logic is materially different (multi-segment qpos sequence)."""

    def __init__(
        self,
        comp_cfg: dict,
        atomic_task_cfgs: dict,         # name -> atomic task cfg from tasks.yaml
        atomic_task_classes: dict,       # name -> task class
        atomic_defaults_cfg: dict,       # tasks.yaml#defaults
        comp_defaults_cfg: dict,         # compositions.yaml#defaults
        physics_cfg: dict,
        eval_only: bool = False,
    ):
        self.cfg = comp_cfg
        self.NAME = comp_cfg["name"]
        self.base_task_names: List[str] = list(comp_cfg["base_tasks"])
        self.atomic_defaults = atomic_defaults_cfg
        self.comp_defaults = comp_defaults_cfg
        self.physics_cfg = physics_cfg
        self.eval_only = eval_only

        # Instantiate one BaseTask per atomic step (for compute_target_qpos)
        self.atomics: List[BaseTask] = []
        for name in self.base_task_names:
            if name not in atomic_task_cfgs:
                raise KeyError(f"Composition '{self.NAME}' references unknown atomic task '{name}'")
            cls = atomic_task_classes[name]
            self.atomics.append(cls(atomic_task_cfgs[name], atomic_defaults_cfg, physics_cfg))

    # ------------------------------------------------------------ generate
    def generate(
        self,
        obj_record: dict,
        seed: int,
        traj_id: str,
        scene,
    ) -> Optional[TrajectoryRecord]:
        rng = random.Random(seed)
        joint_low = obj_record["joint_limit_low"]
        joint_high = obj_record["joint_limit_high"]

        # ---- 1. plan all sub-action targets (chain by overriding qstart)
        sub_targets: List[Tuple[str, float, float, dict]] = []
        prev_end: Optional[float] = None
        for name, atomic in zip(self.base_task_names, self.atomics):
            try:
                qs, qe, info = atomic.compute_target_qpos(joint_low, joint_high, rng)
            except Exception as e:  # noqa: BLE001
                logger.warning("compute_target_qpos failed for sub-task %s: %s", name, e)
                return None
            if prev_end is not None:
                # Override start to chain off previous step's end
                qs = prev_end
                # Recompute info to reflect the override
                info = {**info, "chained_qstart": prev_end}
            # Special case: "open_open_more" -- second open should go further than first
            if name == "open" and prev_end is not None and qs >= qe:
                # The second open has qstart >= qend (because first open already
                # went past where this random sample wants to end). Push qend
                # toward joint_high to ensure further opening.
                qe = max(qe, qs + 0.3 * (joint_high - joint_low))
                qe = min(qe, joint_high)
            sub_targets.append((name, qs, qe, info))
            prev_end = qe

        n_steps = len(sub_targets)

        # ---- 2. sample time budget
        total_s = self.comp_defaults["total_duration_seconds"]
        pre_s = rng.uniform(*self.comp_defaults["pre_settle_range"])
        post_s = rng.uniform(*self.comp_defaults["post_settle_range"])
        inter_s = rng.uniform(*self.comp_defaults["inter_action_settle_range"])
        min_motion = self.comp_defaults.get("min_motion_seconds_per_step", 1.0)

        motion_budget = total_s - pre_s - post_s - inter_s * (n_steps - 1)
        if motion_budget < n_steps * min_motion:
            # Shrink settles to make room for minimum motion budget
            needed = n_steps * min_motion
            shrinkable = total_s - needed - 0.3   # leave a tiny bit for safety
            # Reallocate proportionally to pre / inter / post
            pre_s = max(0.2, shrinkable * 0.15)
            post_s = max(0.5, shrinkable * 0.5)
            inter_s = max(0.2, shrinkable * 0.35 / max(1, n_steps - 1))
            motion_budget = total_s - pre_s - post_s - inter_s * (n_steps - 1)
            if motion_budget < n_steps * 0.3:
                logger.warning("Composition %s has too little motion budget; skipping", self.NAME)
                return None

        motion_per_step = motion_budget / n_steps

        # ---- 3. load articulation
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        urdf_path = str(obj_record["folder"]) + "/mobility.urdf"
        articulation = loader.load(urdf_path)
        if articulation is None:
            logger.warning("Failed to load %s", obj_record["folder"])
            return None

        active_joints = articulation.get_active_joints()
        joint_idx = self._find_joint_index(active_joints, obj_record["joint_name"])
        if joint_idx is None:
            logger.warning("No active joints on %s", obj_record["obj_id"])
            return None

        target_joint = active_joints[joint_idx]
        target_joint.set_drive_property(
            stiffness=1000.0,
            damping=self.physics_cfg.get("default_damping", 0.1) * 100.0,
        )

        # Friction randomization
        physics_params = self._sample_physics(rng)
        for j in active_joints:
            try:
                j.set_friction(physics_params["friction"])
            except Exception:
                pass

        # ---- 4. build full qpos sequence
        fps = self.physics_cfg["fps"]
        n_frames_total = int(round(total_s * fps))
        n_pre = int(round(pre_s * fps))
        n_motion_each = int(round(motion_per_step * fps))
        n_inter = int(round(inter_s * fps))
        # Compute n_post as remainder so total is exact
        n_post = n_frames_total - n_pre - n_steps * n_motion_each - (n_steps - 1) * n_inter
        if n_post < 0:
            # Shave from inter and motion
            shave = -n_post
            n_post = 0
            n_motion_each = max(5, n_motion_each - (shave // n_steps + 1))
        n_post = n_frames_total - n_pre - n_steps * n_motion_each - (n_steps - 1) * n_inter

        qpos_seq = np.empty(n_frames_total, dtype=np.float64)
        sub_action_frame_ranges: List[List[int]] = []
        cursor = 0

        # pre settle: hold first sub-action's qstart
        first_qs = sub_targets[0][1]
        qpos_seq[cursor : cursor + n_pre] = first_qs
        cursor += n_pre

        for i, (name, qs, qe, info) in enumerate(sub_targets):
            # min-jerk motion
            motion_start = cursor
            t_axis = np.arange(n_motion_each) / fps
            s_curve = minimum_jerk_profile(t_axis, T=max(motion_per_step, 1e-3))
            qpos_seq[cursor : cursor + n_motion_each] = qs + s_curve * (qe - qs)
            cursor += n_motion_each
            motion_end = cursor
            sub_action_frame_ranges.append([motion_start, motion_end])

            # inter settle (skip after last sub-action)
            if i < n_steps - 1:
                qpos_seq[cursor : cursor + n_inter] = qe
                cursor += n_inter

        # post settle: hold final qend
        if n_post > 0:
            qpos_seq[cursor : cursor + n_post] = sub_targets[-1][2]
            cursor += n_post

        # Pad/truncate to exact length
        if cursor < n_frames_total:
            qpos_seq[cursor:] = qpos_seq[cursor - 1] if cursor > 0 else first_qs
        elif cursor > n_frames_total:
            qpos_seq = qpos_seq[:n_frames_total]

        # ---- 5. set initial qpos and step physics
        qpos_full = np.zeros(len(active_joints), dtype=np.float64)
        qpos_full[joint_idx] = first_qs
        articulation.set_qpos(qpos_full)

        scene.set_timestep(self.physics_cfg.get("step_dt", 0.0033))
        steps_per_frame = self.physics_cfg.get("steps_per_frame", 10)

        joint_qpos_actual: List[float] = []
        object_pose_world: List[List[float]] = []

        for q_target in qpos_seq:
            target_joint.set_drive_target(float(q_target))
            for _ in range(steps_per_frame):
                scene.step()
            achieved = float(articulation.get_qpos()[joint_idx])
            joint_qpos_actual.append(achieved)
            root_pose = articulation.get_root_pose()
            object_pose_world.append([
                root_pose.p[0], root_pose.p[1], root_pose.p[2],
                root_pose.q[0], root_pose.q[1], root_pose.q[2], root_pose.q[3],
            ])

        # ---- 6. success: every sub-action's end-of-motion within tolerance
        success = True
        for i, (start, end) in enumerate(sub_action_frame_ranges):
            target_qe = sub_targets[i][2]
            achieved = joint_qpos_actual[end - 1]
            joint_range = abs(sub_targets[i][2] - sub_targets[i][1])
            tol = max(0.25 * joint_range, 0.07)
            if abs(achieved - target_qe) > tol:
                success = False
                break

        return TrajectoryRecord(
            traj_id=traj_id,
            obj_id=obj_record["obj_id"],
            obj_category=obj_record["our_category"],
            obj_folder=str(obj_record["folder"]),
            task_name=f"comp:{self.NAME}",      # comp: prefix distinguishes from atomic
            joint_index=obj_record["joint_index"],
            joint_name=obj_record["joint_name"],
            success=success,
            n_frames=len(qpos_seq),
            fps=fps,
            seed=seed,
            joint_qpos=qpos_seq.tolist(),
            joint_qpos_actual=joint_qpos_actual,
            object_pose_world=object_pose_world,
            physics_params=physics_params,
            pre_settle_frames=n_pre,
            motion_frames=n_motion_each * n_steps,   # total motion frames
            post_settle_frames=n_post,
            is_composition=True,
            composition_steps=self.base_task_names,
            sub_action_frame_ranges=sub_action_frame_ranges,
            eval_only=self.eval_only,
            randomization={
                "pre_settle_seconds": pre_s,
                "inter_action_settle_seconds": inter_s,
                "post_settle_seconds": post_s,
                "motion_seconds_per_step": motion_per_step,
                "n_steps": n_steps,
                "sub_action_targets": [
                    {"name": n, "qstart": float(qs), "qend": float(qe), **info}
                    for n, qs, qe, info in sub_targets
                ],
            },
        )

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _find_joint_index(active_joints, joint_name: str) -> Optional[int]:
        if not active_joints:
            return None
        for i, j in enumerate(active_joints):
            if j.name == joint_name:
                return i
        # Fallback to largest range
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
