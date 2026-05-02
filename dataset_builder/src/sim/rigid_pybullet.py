from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RigidRollout:
    poses: np.ndarray
    joint_positions: np.ndarray
    external_force: np.ndarray
    energy_traj: np.ndarray


class PyBulletRigidSimulator:
    def __init__(self, fps: int = 60) -> None:
        self.fps = fps
        try:
            import pybullet as pybullet_module
        except ImportError as exc:
            raise RuntimeError("pybullet is required for rigid simulation") from exc
        self.pb = pybullet_module

    def rollout_joint_action(
        self,
        urdf_path: Path,
        action: str,
        rng: np.random.RandomState,
        frames: int = 60,
    ) -> RigidRollout:
        raise NotImplementedError(
            f"Rigid rollout for {action} from {urdf_path} is implemented in Phase 1"
        )
