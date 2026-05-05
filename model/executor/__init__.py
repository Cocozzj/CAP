"""
Executor  —  Physics-Aware Executor for the CAP system.

Public API:
  - Executor      : top-level orchestrator (apply_token / apply_sequence / transfer_object)

Subpackage exports (advanced use):
  - rigid    : RigidTransform, DiscreteRotationGroup, EquivariantResidual
  - deform   : DeformSim, RhoParser, VelocityFieldTFN, PhysicsRouter, BlackBoxFallback
  - physics  : PhysicsParams, DeformResult, DiffDeformer, RigidContactBackend,
               ShapeMatchingPBD, MPMBackend, verlet_step

Note:
  - ``SceneState`` lives in ``model/utils/utils.py``; import directly from there.
  - Appearance helpers ``rotate_sh_l1`` / ``compensate_appearance`` are defined
    inline inside ``deform/sim.py`` and are not part of the public API.  Import
    them from ``model.executor.deform.sim`` if you really need them.
"""

from .Executor import Executor

__all__ = ["Executor"]
