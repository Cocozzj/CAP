from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeformableRollout:
    particles_pos: np.ndarray
    particles_vel: np.ndarray
    external_force: np.ndarray
    energy_traj: np.ndarray
    volume_traj: np.ndarray


class TaichiDeformableSimulator:
    def __init__(self, grid_size: int = 64, dt: float = 1.0 / 120.0, substeps: int = 4) -> None:
        self.grid_size = grid_size
        self.dt = dt
        self.substeps = substeps
        try:
            import taichi as taichi_module
        except ImportError as exc:
            raise RuntimeError("taichi is required for deformable simulation") from exc
        self.ti = taichi_module

    def rollout(self, action: str, rng: np.random.RandomState, frames: int = 60) -> DeformableRollout:
        raise NotImplementedError(f"Deformable rollout for {action} is implemented in Phase 3")
