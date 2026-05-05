"""
physics/base.py — Foundation for differentiable physics backends.

Contains:
  - PhysicsParams        : dataclass holding decoded physics parameters
  - DeformResult         : dataclass returned by every backend's simulate()
  - DiffDeformer         : abstract base class for physics backends
  - verlet_step          : symmetric (Stoermer-Verlet) integrator

SPD covariance utilities (``cov_to_log_euclidean`` / ``log_euclidean_to_cov``
/ ``project_spd``) live in ``model/utils/utils.py`` since they're shared by
the canonical-frame code, the Encoder, and the loss suite.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


# ═══════════════════════════════════════════════════════════════════════════
# §1  Dataclasses (PhysicsParams + DeformResult)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PhysicsParams:
    """Structured physics parameters decoded from ρ by RhoParser.

    All tensor fields have shape [M, ...] where M = B·K (flattened object dim).
    `n_iters` and `n_substeps` are integration-loop hyperparameters set by
    RhoParser at construction time (constant across the batch — they are NOT
    learned per-object, both because integer counts can't be back-propped and
    because per-step CUDA→CPU sync via .item() was a measurable training cost).
    Defaults below are aligned with RhoParser's defaults.
    """
    youngs_modulus:  torch.Tensor      # [M, 1]   Young's modulus E
    poisson_ratio:   torch.Tensor      # [M, 1]   Poisson ratio ν ∈ (0, 0.5)
    density:         torch.Tensor      # [M, 1]   mass density ρ_m
    ext_force:       torch.Tensor      # [M, 3]   external force vector
    friction_coeff:  torch.Tensor      # [M, 1]   Coulomb friction μ ∈ [0, 1]
    damping:         torch.Tensor      # [M, 1]   velocity damping ∈ [0, 1]
    dt:              torch.Tensor      # [M, 1]   per-object timestep
    n_iters:         int = 4           # PBD iteration count   (RhoParser default)
    n_substeps:      int = 2           # Verlet substep count  (RhoParser default)


@dataclass
class DeformResult:
    """Return type shared by all backends."""
    delta_mu:  torch.Tensor    # [M, N, 3]      position displacement
    delta_cov: torch.Tensor    # [M, N, 3, 3]   covariance increment (linear or log-Euclidean)
    R_loc:    torch.Tensor     # [M, N, 3, 3]   local rotation (for SH compensation)
    J:        torch.Tensor     # [M, N]         volume Jacobian (for opacity/scale compensation)


# ═══════════════════════════════════════════════════════════════════════════
# §2  Abstract base class for differentiable physics backends
# ═══════════════════════════════════════════════════════════════════════════

class DiffDeformer(abc.ABC):
    """Abstract differentiable deformer.

    Every concrete backend must implement ``simulate``.

    The ``mask`` argument is OPTIONAL but recommended:
      - When provided ([M, N] bool), padded positions (mask=False) are
        excluded from aggregations such as centre-of-mass and SVD shape
        matching.  This is critical for correct physics on padded SceneState.
      - When None, all N positions are treated as real (legacy behaviour).
    """

    @abc.abstractmethod
    def simulate(
        self,
        positions: torch.Tensor,                   # [M, N, 3]  current Gaussian centres
        params: PhysicsParams,
        mask: Optional[torch.Tensor] = None,       # [M, N] bool, True = real
    ) -> DeformResult:
        """Run physics simulation and return deformation deltas."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# §3  Symmetric Verlet integrator (PDF §1.3 — time-reversible)
# ═══════════════════════════════════════════════════════════════════════════

def verlet_step(
    x: torch.Tensor,        # [..., N, 3]  positions
    v: torch.Tensor,        # [..., N, 3]  velocities
    accel_fn,               # callable(x) → a [..., N, 3]
    dt,                     # tensor of any rank ≤ x.ndim, OR Python scalar
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stoermer-Verlet (velocity form), symmetric and time-reversible:
        v_{1/2}  = v  + (dt/2) a(x)
        x'       = x  + dt · v_{1/2}
        v'       = v_{1/2}  + (dt/2) a(x')

    Broadcasting:
        ``dt`` may be a Python float, a 0-d tensor, or any tensor whose rank
        is ≤ x.ndim.  Trailing singleton dims are appended as needed so the
        product ``dt * a`` broadcasts cleanly over (N, 3).
    """
    if isinstance(dt, torch.Tensor):
        # Append trailing singleton dims until dt rank matches x rank.
        # Robust to per-object [M, 1], scalar 0-d, or any future rank.
        while dt.dim() < x.dim():
            dt = dt.unsqueeze(-1)
    a0 = accel_fn(x)
    v_half = v + 0.5 * dt * a0
    x_new  = x + dt * v_half
    a1 = accel_fn(x_new)
    v_new  = v_half + 0.5 * dt * a1
    return x_new, v_new
