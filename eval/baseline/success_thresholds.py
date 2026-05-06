"""Per-task success-rate thresholds.

A trajectory is judged successful if the **moving part** reached
sufficient configuration change toward the task target:

  • For revolute joints (open/close/rotate/fold/spin): min angle change
    of ``angle_deg``.
  • For prismatic joints (push/pull/slide): min translation of
    ``distance_m``.

Comparison direction:
  ``"toward_target"`` — pred must reach within ``tolerance_frac`` of GT
                         target_fraction (default mode for most tasks).
  ``"absolute"`` — just measures absolute change magnitude regardless
                    of direction (looser).

A simple baseline policy table — values are calibrated against PartNet
joint limits (typical ~1–2 rad, ~0.1–0.3 m) and what humans judge as
"clearly performed" the action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TaskThreshold:
    angle_rad:        Optional[float] = None    # revolute min change
    distance_m:       Optional[float] = None    # prismatic min change
    direction:        str = "toward_target"     # see module docstring
    tolerance_frac:   float = 0.30              # "within 30 % of target"
    description:      str = ""


# Map of task_name → threshold
# Composite tasks (``comp:open_close``, etc.) decompose to their parts;
# all sub-task thresholds must be satisfied for the composite to count.
ATOMIC_THRESHOLDS: Dict[str, TaskThreshold] = {
    "open":   TaskThreshold(angle_rad=0.523,  description="≥ 30°"),
    "close":  TaskThreshold(angle_rad=0.523,  description="≥ 30°"),
    "rotate": TaskThreshold(angle_rad=0.523,  description="≥ 30°"),
    "fold":   TaskThreshold(angle_rad=1.047,  description="≥ 60°"),
    "spin":   TaskThreshold(angle_rad=0.785,  description="≥ 45°"),
    "push":   TaskThreshold(distance_m=0.05,  description="≥ 5 cm"),
    "pull":   TaskThreshold(distance_m=0.05,  description="≥ 5 cm"),
    "slide":  TaskThreshold(distance_m=0.05,  description="≥ 5 cm"),
    "lift":   TaskThreshold(distance_m=0.10,  description="≥ 10 cm"),
}


def threshold_for(task_name: str) -> Optional[TaskThreshold]:
    """Look up threshold; returns None for unknown task verbs."""
    if task_name in ATOMIC_THRESHOLDS:
        return ATOMIC_THRESHOLDS[task_name]
    # Composite — pick the strictest atomic verb mentioned.
    if task_name.startswith("comp:"):
        verbs = task_name.split(":", 1)[1].split("_")
        thresholds = [ATOMIC_THRESHOLDS[v] for v in verbs if v in ATOMIC_THRESHOLDS]
        if not thresholds:
            return None
        # For composite, take the *first* (= sub-task with the smallest required
        # change for the FIRST half of the trajectory; we evaluate per-stage
        # separately in success_composite()).
        return thresholds[0]
    return None


def is_revolute_task(task_name: str) -> bool:
    """Heuristic: which atomic verbs target revolute joints."""
    revolute_verbs = {"open", "close", "rotate", "fold", "spin"}
    if task_name in revolute_verbs:
        return True
    if task_name.startswith("comp:"):
        verbs = task_name.split(":", 1)[1].split("_")
        return any(v in revolute_verbs for v in verbs)
    return False


__all__ = [
    "TaskThreshold",
    "ATOMIC_THRESHOLDS",
    "threshold_for",
    "is_revolute_task",
]
