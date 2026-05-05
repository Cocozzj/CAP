"""
physics/rigid_contact.py — Rigid-body Newton dynamics + Coulomb friction.

Treats each object as a rigid body with a centre-of-mass (COM).  Applies:
  - external force from PhysicsParams.ext_force
  - gravity along -z
  - ground contact at z = ground_z (with penetration correction)
  - Coulomb friction damping horizontal velocity on contact

Mask-aware:
  - COM computed via masked mean → padded zeros do NOT bias the dynamics
  - Per-Gaussian displacement equals the COM displacement (rigid → all same)

Output convention (DeformResult):
  - delta_mu  : COM displacement, broadcast to all N (real and padded)
                 (padded positions get the same delta but are filtered later by mask)
  - delta_cov : 0  (rigid body — no shape change in this backend)
  - R_loc     : I  (no local rotation here; rigid R comes from the rigid branch)
  - J         : 1  (no volume change)
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import DiffDeformer, DeformResult, PhysicsParams, verlet_step
from ...utils import masked_mean


class RigidContactBackend(DiffDeformer):
    """Newton dynamics on object COM + ground contact + Coulomb friction."""

    def __init__(self, gravity: float = -9.81, ground_z: float = 0.0) -> None:
        self.gravity = gravity
        self.ground_z = ground_z

    def simulate(
        self,
        positions: torch.Tensor,                   # [M, N, 3]
        params: PhysicsParams,
        mask: Optional[torch.Tensor] = None,       # [M, N] bool — True = real
    ) -> DeformResult:
        M, N, _ = positions.shape
        dev = positions.device

        # ── Centre of mass (mask-aware: padded zeros excluded) ────────
        if mask is not None:
            com = masked_mean(positions, mask, dim=1, keepdim=True)   # [M, 1, 3]
        else:
            com = positions.mean(dim=1, keepdim=True)

        # ── Bounding extent for mass estimate (also mask-aware) ───────
        # Use real positions only; padded zeros would otherwise inflate extent.
        if mask is not None:
            mask_f = mask.float().unsqueeze(-1)                       # [M, N, 1]
            # Set padded positions to COM so they don't expand min/max
            pos_for_extent = positions * mask_f + com * (1 - mask_f)
        else:
            pos_for_extent = positions

        extent = (pos_for_extent.max(1).values - pos_for_extent.min(1).values).clamp(min=1e-4)
        volume = extent.prod(-1, keepdim=True)                        # [M, 1]
        mass = (params.density * volume).clamp(min=1e-3)              # [M, 1]

        # ── Constant acceleration: gravity + external force / mass ────
        grav = torch.zeros(M, 1, 3, device=dev, dtype=positions.dtype)
        grav[..., 2] = self.gravity
        f_ext = params.ext_force.unsqueeze(1)                         # [M, 1, 3]
        accel_const = grav + f_ext / mass.unsqueeze(-1)

        def accel_fn(x):
            return accel_const.expand_as(x)

        # ── Integrate COM with Verlet substeps ────────────────────────
        v0 = torch.zeros_like(com)
        sub_dt = params.dt / max(params.n_substeps, 1)
        com_new = com
        for _ in range(params.n_substeps):
            com_new, v0 = verlet_step(com_new, v0, accel_fn, sub_dt)

        # ── Ground contact: snap penetrating COM back to ground ──────
        below = (com_new[..., 2:3] < self.ground_z).float()           # [M, 1, 1]
        penetration = (self.ground_z - com_new[..., 2:3]).clamp(min=0)
        com_new = com_new + below * torch.cat([
            torch.zeros(M, 1, 2, device=dev, dtype=positions.dtype),
            penetration
        ], -1)

        # ── Coulomb friction: on contact, horizontal motion is damped ─
        # Friction-induced displacement reduction:
        #   Δx_horiz_after_friction = Δx_horiz · (1 - μ · contact)
        # where contact ∈ {0, 1} is the ground-contact indicator.
        # This shrinks the realised horizontal COM displacement, which is
        # what physically happens when the contact friction force partially
        # cancels horizontal momentum during the substep.
        # (Previously friction was applied to a discarded local velocity
        # variable, so friction_coeff had no effect on the output — broke
        # PDF f07d2c0a's 跨摩擦泛化 / 反事实摩擦验证.)
        delta_com = com_new - com                                     # [M, 1, 3]
        friction_factor = (1 - below * params.friction_coeff.unsqueeze(-1))  # [M, 1, 1]
        delta_com = torch.cat([
            delta_com[..., :2] * friction_factor,                     # horizontal damped
            delta_com[..., 2:3],                                      # vertical unchanged
        ], dim=-1)

        # ── COM displacement → per-Gaussian displacement (rigid: uniform) ─
        delta_mu = delta_com.expand(M, N, 3)

        # ── Rigid → no local rotation, no volume change ──────────────
        R_loc = torch.eye(3, device=dev, dtype=positions.dtype).expand(M, N, 3, 3)
        J = torch.ones(M, N, device=dev, dtype=positions.dtype)
        delta_cov = torch.zeros(M, N, 3, 3, device=dev, dtype=positions.dtype)

        return DeformResult(delta_mu=delta_mu, delta_cov=delta_cov,
                            R_loc=R_loc, J=J)
