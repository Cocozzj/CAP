"""Task templates: each defines how to drive an articulated object's joint
through a parameterized motion to realize a high-level action verb.

Tasks operate at the joint level (using SAPIEN's drive controllers). They
do *not* require an end-effector or grasping — for a synthetic dataset where
we only need the joint to move predictably, that is sufficient and avoids
heavy dependencies on motion-planning libraries.
"""

from .base import BaseTask, TrajectoryRecord
from .composite import CompositeTask
from .open_close import OpenTask, CloseTask
from .pull_push import PullTask, PushTask
from .rotate import RotateTask
from .soft import FoldTask, PourTask, SoftBaseTask, SqueezeTask

ALL_TASK_CLASSES = {
    "open":    OpenTask,
    "close":   CloseTask,
    "pull":    PullTask,
    "push":    PushTask,
    "rotate":  RotateTask,
    "squeeze": SqueezeTask,
    "fold":    FoldTask,
    "pour":    PourTask,
}


def get_task_class(name: str) -> type[BaseTask]:
    if name not in ALL_TASK_CLASSES:
        raise KeyError(f"Unknown task '{name}'. Known: {list(ALL_TASK_CLASSES)}")
    return ALL_TASK_CLASSES[name]


__all__ = [
    "BaseTask",
    "CompositeTask",
    "SoftBaseTask",
    "TrajectoryRecord",
    "OpenTask",
    "CloseTask",
    "PullTask",
    "PushTask",
    "RotateTask",
    "SqueezeTask",
    "FoldTask",
    "PourTask",
    "ALL_TASK_CLASSES",
    "get_task_class",
]
