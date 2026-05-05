"""Physics backends for the Executor's deformation branch.

Exports:
  - PhysicsParams, DeformResult, DiffDeformer  (base contract)
  - RigidContactBackend, ShapeMatchingPBD, MPMBackend  (3 concrete backends)
  - verlet_step  (symmetric integrator)

SPD covariance utilities (cov_to_log_euclidean, log_euclidean_to_cov,
project_spd) live in ``model/utils/utils.py``; import from there.
"""

from .base import (
    PhysicsParams,
    DeformResult,
    DiffDeformer,
    verlet_step,
)
from .rigid_contact import RigidContactBackend
from .pbd import ShapeMatchingPBD
from .mpm import MPMBackend

__all__ = [
    # Base contract
    "PhysicsParams",
    "DeformResult",
    "DiffDeformer",
    # Backends
    "RigidContactBackend",
    "ShapeMatchingPBD",
    "MPMBackend",
    # Utilities
    "verlet_step",
]
