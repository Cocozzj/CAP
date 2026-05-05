"""
deform/rho_parser.py — Decode the deformation vector ρ into structured PhysicsParams.

Maps the flat ρ vector (per-object, dimension rho_dim) to physical quantities:
  Young's modulus E, Poisson ratio ν, density, external force, friction,
  velocity damping, and a learnable per-object timestep dt.

Optional task conditioning:
  ρ_eff = (1 + tanh(A)) · ρ + b      where (A, b) depend on task_context

Spectral normalisation on every Linear bounds the Lipschitz constant of the
parser (PDF §1.4 — keeps physics gradients well-behaved).

Used by:
  - executor/deform/sim.py  (DeformSim calls RhoParser then feeds PhysicsParams
                              to the PhysicsRouter and individual backends)

References:
  - Physics Plugin PDF §1.3  (ρ → physics parameter mapping)
  - Main Proposal §4.2.3(B)  (per-object material decoding)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics import PhysicsParams


class RhoParser(nn.Module):
    """Decode ρ ∈ ℝ^{rho_dim} into structured PhysicsParams."""

    def __init__(
        self,
        rho_dim: int = 16,
        task_dim: int = 0,
        use_spectral_norm: bool = True,
        n_iters: int = 5,
        n_substeps: int = 2,
    ) -> None:
        """
        Args:
            rho_dim:           dim of ρ vector
            task_dim:          dim of optional task conditioning (0 = off)
            use_spectral_norm: bound Lipschitz constant (PDF §1.4)
            n_iters:           PBD iteration count.  Default 5 follows PDF main-
                               proposal hyperparam table; this is a constant —
                               was learned earlier but that triggers per-step
                               CUDA→CPU sync via .item().
            n_substeps:        Verlet substep count (constant for the same reason)
        """
        super().__init__()
        self.rho_dim    = rho_dim
        self.n_iters    = int(n_iters)
        self.n_substeps = int(n_substeps)

        def _maybe_sn(layer: nn.Linear) -> nn.Linear:
            """Apply spectral normalisation to bound Lipschitz constant (PDF §1.4)."""
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # Heads: ρ → individual physics quantities
        # Spectral norm constrains  ‖∂params/∂ρ‖ ≤ 1 per layer
        self.head_youngs   = _maybe_sn(nn.Linear(rho_dim, 1))
        self.head_poisson  = _maybe_sn(nn.Linear(rho_dim, 1))
        self.head_density  = _maybe_sn(nn.Linear(rho_dim, 1))
        self.head_force    = _maybe_sn(nn.Linear(rho_dim, 3))
        self.head_friction = _maybe_sn(nn.Linear(rho_dim, 1))
        self.head_damping  = _maybe_sn(nn.Linear(rho_dim, 1))
        # dt only (n_iters / n_substeps are now constructor params)
        self.head_dt       = _maybe_sn(nn.Linear(rho_dim, 1))

        # Optional task conditioning  (affine modulation of ρ)
        self.has_task = task_dim > 0
        if self.has_task:
            self.task_affine = _maybe_sn(nn.Linear(task_dim, rho_dim * 2))

    def forward(
        self,
        rho: torch.Tensor,                          # [M, rho_dim]
        task_context: Optional[torch.Tensor] = None,  # [M, task_dim] or None
    ) -> PhysicsParams:
        """Decode ρ → PhysicsParams (with optional task conditioning)."""

        # ── Optional task conditioning: ρ_eff = (1 + tanh(A)) · ρ + b ──
        if self.has_task and task_context is not None:
            ab = self.task_affine(task_context)             # [M, 2 * rho_dim]
            A, b = ab.chunk(2, dim=-1)
            rho = (1 + A.tanh()) * rho + b

        # NaN guard (rho can come from upstream physics propagation)
        rho = torch.nan_to_num(rho, nan=0.0, posinf=1e4, neginf=-1e4)

        # ── Material parameters with appropriate activations ──────────
        youngs   = F.softplus(self.head_youngs(rho))                # > 0
        poisson  = self.head_poisson(rho).sigmoid() * 0.499         # (0, 0.5)
        density  = F.softplus(self.head_density(rho)) + 0.1         # > 0.1
        force    = self.head_force(rho)
        friction = self.head_friction(rho).sigmoid()                # [0, 1]
        damping  = self.head_damping(rho).sigmoid() * 0.5           # [0, 0.5]

        # ── Timestep (per-object positive scalar) ────────────────────
        dt_raw = self.head_dt(rho)
        dt_raw = torch.nan_to_num(dt_raw, nan=0.0, posinf=1e4, neginf=-1e4)
        dt     = F.softplus(dt_raw) * 0.01 + 1e-4   # ~1e-4 .. ~1e-2 seconds

        # n_iters / n_substeps are constants (constructor params) — see __init__.
        return PhysicsParams(
            youngs_modulus=youngs,
            poisson_ratio=poisson,
            density=density,
            ext_force=force,
            friction_coeff=friction,
            damping=damping,
            dt=dt,
            n_iters=self.n_iters,
            n_substeps=self.n_substeps,
        )
