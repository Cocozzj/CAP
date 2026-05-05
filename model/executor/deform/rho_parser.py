"""
deform/rho_parser.py — Slice the 9-dim deformation vector ρ into named PhysicsParams.

Following Physics Plugin PDF §1.3, ρ is defined as a structured tuple of
named physical parameters (material / force / contact / boundary), NOT a
latent embedding.  This module is therefore a thin "named slicer" that
applies range-correct activations to each slot — no learnable weights per
slot, just the activation.

Layout (rho_dim = 9):
    rho[..., 0]    Young's modulus       E         > 0      (softplus)
    rho[..., 1]    Poisson ratio         ν         (0, 0.499)  (sigmoid * 0.499)
    rho[..., 2]    density               ρ_m       > 0.1    (softplus + 0.1)
    rho[..., 3:6]  external force        F ∈ ℝ³    unbounded
    rho[..., 6]    Coulomb friction      μ         [0, 1]   (sigmoid)
    rho[..., 7]    velocity damping      ζ         [0, 0.5] (sigmoid * 0.5)
    rho[..., 8]    per-object timestep   dt        ~1e-4..1e-2 s  (softplus * 0.01 + 1e-4)

Each VQ codebook entry is therefore a complete material/force "recipe" —
directly inspectable, no decoding needed.

Optional task conditioning:
  ρ_eff = (1 + tanh(A)) · ρ + b      where (A, b) = task_affine(task_context)

Used by:
  - executor/deform/sim.py  (DeformSim calls RhoParser then feeds PhysicsParams
                              to the PhysicsRouter and individual backends)

References:
  - Physics Plugin PDF §1.3  (ρ = structured named physics tuple)
  - Main Proposal §2.3        (executor consumes ρ to drive simulator)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics import PhysicsParams


# Slot indices into the 9-dim ρ vector — single source of truth.
RHO_DIM      = 9
SLOT_YOUNGS  = 0
SLOT_POISSON = 1
SLOT_DENSITY = 2
SLOT_FORCE   = slice(3, 6)
SLOT_FRICTION= 6
SLOT_DAMPING = 7
SLOT_DT      = 8


class RhoParser(nn.Module):
    """Slice ρ ∈ ℝ⁹ into structured PhysicsParams via per-slot activations.

    Has NO per-slot Linear weights — each slot just gets its range-correct
    activation function.  The only learnable params are inside the optional
    ``task_affine`` modulator.
    """

    def __init__(
        self,
        rho_dim: int = RHO_DIM,
        task_dim: int = 0,
        use_spectral_norm: bool = True,
        n_iters: int = 5,
        n_substeps: int = 2,
    ) -> None:
        """
        Args:
            rho_dim:           dim of ρ vector (must be 9 — kept as kwarg for
                               backward-compat config plumbing; raises otherwise)
            task_dim:          dim of optional task conditioning (0 = off)
            use_spectral_norm: bound Lipschitz constant of the optional
                               task_affine layer (PDF §1.4)
            n_iters:           PBD iteration count (constant, see __init__ note)
            n_substeps:        Verlet substep count (constant)
        """
        super().__init__()
        if rho_dim != RHO_DIM:
            raise ValueError(
                f"RhoParser is hard-wired to rho_dim={RHO_DIM} (named slots: "
                f"E, ν, ρ_m, F[3], μ, damping, dt).  Got rho_dim={rho_dim}; "
                f"set deformation.dim={RHO_DIM} in configs/config.yaml."
            )
        self.rho_dim    = RHO_DIM
        self.n_iters    = int(n_iters)
        self.n_substeps = int(n_substeps)

        # Task conditioning is the ONLY learnable part — per-slot activations
        # are weight-free.  Spectral norm caps ‖∂ρ_eff/∂task‖ ≤ 1.
        self.has_task = task_dim > 0
        if self.has_task:
            lin = nn.Linear(task_dim, RHO_DIM * 2)
            self.task_affine = nn.utils.spectral_norm(lin) if use_spectral_norm else lin

    def forward(
        self,
        rho: torch.Tensor,                            # [M, 9]
        task_context: Optional[torch.Tensor] = None,  # [M, task_dim] or None
    ) -> PhysicsParams:
        """Slice ρ → PhysicsParams (with optional task conditioning).

        Args:
            rho:           [M, 9]  per-object deformation vector with named slots
            task_context:  [M, task_dim] or None
        """
        # ── Optional task conditioning: ρ_eff = (1 + tanh(A)) · ρ + b ──
        if self.has_task and task_context is not None:
            ab = self.task_affine(task_context)               # [M, 18]
            A, b = ab.chunk(2, dim=-1)                        # each [M, 9]
            rho = (1.0 + A.tanh()) * rho + b

        # NaN guard (rho can come from upstream physics propagation)
        rho = torch.nan_to_num(rho, nan=0.0, posinf=1e4, neginf=-1e4)

        # ── Per-slot named slicing + range-correct activation ─────────
        youngs   = F.softplus(rho[..., SLOT_YOUNGS]).unsqueeze(-1)         # [M, 1]  > 0
        poisson  = rho[..., SLOT_POISSON].sigmoid().unsqueeze(-1) * 0.499  # [M, 1]  (0, 0.499)
        density  = (F.softplus(rho[..., SLOT_DENSITY]) + 0.1).unsqueeze(-1)# [M, 1]  > 0.1
        force    = rho[..., SLOT_FORCE]                                    # [M, 3]  unbounded
        friction = rho[..., SLOT_FRICTION].sigmoid().unsqueeze(-1)         # [M, 1]  [0, 1]
        damping  = rho[..., SLOT_DAMPING].sigmoid().unsqueeze(-1) * 0.5    # [M, 1]  [0, 0.5]

        # dt range [1e-4, ~1e-2] s = [0.1ms, 10ms] — tighter than PDF's nominal
        # 1/30 s, follows standard PBD practice (Müller 2007: dt ≤ 10ms keeps
        # the 5-iter constraint solver stable across rubber/foam/cloth).
        dt = (F.softplus(rho[..., SLOT_DT]) * 0.01 + 1e-4).unsqueeze(-1)   # [M, 1]

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
