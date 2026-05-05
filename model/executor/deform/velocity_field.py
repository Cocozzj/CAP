"""
deform/velocity_field.py — Learned equivariant velocity field v_ρ(x).

Architecture (per object, per Gaussian):
    rho_embed → radial weights w_j(ρ), strength s(ρ)
    v(x) = s(ρ) · Σ_j  w_j  ·  (x - c_j) / ‖x - c_j‖
where {c_j} are virtual source points placed near the object's COM.

The velocity field is then integrated with symmetric Verlet to produce a
displacement Δμ that is added to the physics-backend output.

Mask-aware:
  - COM computed via masked_mean → padded zeros do NOT bias source positions
  - The displacement Δμ is computed for ALL particles; padded ones get
    garbage values that are filtered out downstream by mask

References:
  - Physics Plugin PDF §1.3  (velocity field + Verlet integration)
  - Main Proposal §4.2.3(B)  (continuous deformation between rigid steps)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..physics import verlet_step
from ...utils import masked_mean


class VelocityFieldTFN(nn.Module):
    """Learned equivariant velocity field, integrated with Verlet."""

    def __init__(
        self,
        rho_dim: int = 16,
        n_sources: int = 4,
        hidden: int = 32,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.n_sources = n_sources
        # Source point offsets, learned around the COM
        self.source_offsets = nn.Parameter(torch.randn(n_sources, 3) * 0.1)

        def _maybe_sn(layer: nn.Linear) -> nn.Linear:
            """Spectral normalisation to bound Lipschitz constant (PDF §1.4)."""
            return nn.utils.spectral_norm(layer) if use_spectral_norm else layer

        # ρ → per-source radial weight
        self.rho_to_weights = nn.Sequential(
            _maybe_sn(nn.Linear(rho_dim, hidden)), nn.GELU(),
            _maybe_sn(nn.Linear(hidden, n_sources)),
        )
        # ρ → overall strength (-1 to 1 via tanh)
        self.rho_to_strength = nn.Sequential(
            _maybe_sn(nn.Linear(rho_dim, hidden)), nn.GELU(),
            _maybe_sn(nn.Linear(hidden, 1)), nn.Tanh(),
        )

    # ──────────────────────────────────────────────────────────────────
    # Velocity evaluation
    # ──────────────────────────────────────────────────────────────────

    def compute_velocity(
        self,
        x: torch.Tensor,                           # [M, N, 3]   particle positions
        rho: torch.Tensor,                         # [M, rho_dim]
        mask: Optional[torch.Tensor] = None,       # [M, N] bool — affects COM only
    ) -> torch.Tensor:
        """Evaluate v_ρ(x) at each Gaussian position.  → [M, N, 3]."""
        M, N, _ = x.shape

        # ── COM (mask-aware) → place virtual sources around it ───────
        if mask is not None:
            com = masked_mean(x, mask, dim=1, keepdim=True)            # [M, 1, 3]
        else:
            com = x.mean(1, keepdim=True)
        centres = com + self.source_offsets[None, :, :]                # [M, J, 3]

        # ── Per-source weights / strength from ρ ─────────────────────
        w = self.rho_to_weights(rho)                                    # [M, J]
        s = self.rho_to_strength(rho)                                   # [M, 1]

        # ── Radial basis: v = s · Σ_j  w_j · (x - c_j) / ‖x - c_j‖ ──
        diff = x[:, :, None, :] - centres[:, None, :, :]                # [M, N, J, 3]
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        basis = diff / dist                                             # normalised direction

        v = (w[:, None, :, None] * basis).sum(dim=2)                    # [M, N, 3]
        v = s[:, :, None] * v                                           # scale by strength
        return v

    # ──────────────────────────────────────────────────────────────────
    # Forward (Verlet integration)
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        positions: torch.Tensor,                  # [M, N, 3]
        rho: torch.Tensor,                        # [M, rho_dim]
        dt: torch.Tensor,                         # [M, 1]
        n_substeps: int = 2,
        mask: Optional[torch.Tensor] = None,      # [M, N] bool — passed to COM
    ) -> torch.Tensor:
        """Integrate the learned field as an acceleration with Verlet.

        The network learns to compensate for the "velocity ↔ acceleration"
        semantic mismatch (we feed v_ρ as accel into Verlet for symmetry).

        Returns:
            displacement Δμ:  [M, N, 3]
        """
        x = positions.clone()
        v0 = torch.zeros_like(x)

        def accel_fn(xq):
            return self.compute_velocity(xq, rho, mask=mask)

        sub_dt = dt / max(n_substeps, 1)
        for _ in range(max(n_substeps, 1)):
            x, v0 = verlet_step(x, v0, accel_fn, sub_dt)

        return x - positions
