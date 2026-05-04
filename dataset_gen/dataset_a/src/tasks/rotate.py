"""Rotate: revolute joint motion for faucets / scissors / similar."""

from __future__ import annotations

import random
from typing import Tuple

from .base import BaseTask


class RotateTask(BaseTask):
    NAME = "rotate"

    def compute_target_qpos(
        self,
        joint_low: float,
        joint_high: float,
        rng: random.Random,
    ) -> Tuple[float, float, dict]:
        target_frac_range = self.cfg.get("target_fraction_range", [0.5, 1.0])
        target_frac = rng.uniform(*target_frac_range)
        # Rotate from low to a fraction toward high
        qstart = joint_low
        qend = joint_low + target_frac * (joint_high - joint_low)
        return qstart, qend, {"target_fraction": target_frac, "direction": "rotate"}
