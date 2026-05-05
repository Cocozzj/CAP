"""Non-rigid deformation branch of the Executor.

Exports:
  - DeformSim          : main orchestrator (RhoParser → VFieldTFN → Router → appearance)
  - RhoParser          : ρ → PhysicsParams
  - VelocityFieldTFN   : learned v_ρ(x) + Verlet integration
  - PhysicsRouter      : multi-backend Gumbel-softmax dispatch
  - BlackBoxFallback   : Δscale + Δopacity when physics is disabled
"""

from .sim import DeformSim
from .rho_parser import RhoParser
from .velocity_field import VelocityFieldTFN
from .router import PhysicsRouter
from .fallback import BlackBoxFallback

__all__ = [
    "DeformSim",
    "RhoParser",
    "VelocityFieldTFN",
    "PhysicsRouter",
    "BlackBoxFallback",
]
