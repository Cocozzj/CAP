"""Pull / push: prismatic joint motion (drawers, racks, sashes)."""

from __future__ import annotations

import random
from typing import Tuple

from .base import BaseTask


class PullTask(BaseTask):
    NAME = "pull"

    def compute_target_qpos(
        self,
        joint_low: float,
        joint_high: float,
        rng: random.Random,
    ) -> Tuple[float, float, dict]:
        target_frac_range = self.cfg.get("target_fraction_range", [0.6, 1.0])
        target_frac = rng.uniform(*target_frac_range)
        qstart = joint_low
        qend = joint_low + target_frac * (joint_high - joint_low)
        return qstart, qend, {"target_fraction": target_frac, "direction": "extend"}


class PushTask(BaseTask):
    NAME = "push"

    def compute_target_qpos(
        self,
        joint_low: float,
        joint_high: float,
        rng: random.Random,
    ) -> Tuple[float, float, dict]:
        start_frac_range = self.cfg.get("start_from_open_fraction", [0.6, 1.0])
        start_frac = rng.uniform(*start_frac_range)
        target_frac = self.cfg.get("target_fraction", 0.0)
        qstart = joint_low + start_frac * (joint_high - joint_low)
        qend = joint_low + target_frac * (joint_high - joint_low)
        return qstart, qend, {
            "start_fraction": start_frac,
            "target_fraction": target_frac,
            "direction": "retract",
        }
