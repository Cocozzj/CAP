"""Rigid-body branch of the Executor.

Exports:
  - RigidTransform           : main rigid SE(3) transform module
  - DiscreteRotationGroup    : cube24 rotation lookup utility (optional use)
  - EquivariantResidual      : 2-layer TFN for quantisation residual
  - build_discrete_rotation_group : standalone cube24 builder (no nn.Module)
"""

from .transform import RigidTransform
from .rotation_group import DiscreteRotationGroup, build_discrete_rotation_group
from .tfn_residual import EquivariantResidual

__all__ = [
    "RigidTransform",
    "DiscreteRotationGroup",
    "build_discrete_rotation_group",
    "EquivariantResidual",
]
