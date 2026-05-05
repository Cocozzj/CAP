"""
rigid/transform.py — Rigid SE(3) transform branch of the Executor.

Composes total rotation  R = exp(ξ) · R_h  (continuous residual × discrete /
prior rotation matrix supplied by Encoder.head_h) and translation  t = ℓ,
then applies SE(3) to per-object Gaussians using the row-vector convention:

    μ' = μ R^T + t            (positions)
    Σ' = R Σ R^T              (covariances)

Optionally adds a TFN-based equivariant residual (Δμ, Δs) for quantisation
compensation.

Tensor layout:  [B, K, N, ...]   (B batches × K objects × N Gaussians).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .rotation_group import DiscreteRotationGroup  # kept available for utilities
from .tfn_residual import EquivariantResidual

from ...utils import exp_so3

# ═══════════════════════════════════════════════════════════════════════════
# §1  RigidTransform — full rigid-body update
# ═══════════════════════════════════════════════════════════════════════════

class RigidTransform(nn.Module):
    """Apply rigid SE(3) to per-object Gaussians.

    Given parsed (ℓ, R_h, ξ), composes  R = exp(ξ) · R_h  and applies:
        μ' = μ R^T + t
        Σ' = R Σ R^T
    SH is left untouched here (rotated downstream in the deform/appearance
    stage with the local rotation R_loc, or in the no-physics fallback path).

    Optional TFN residual adds per-Gaussian (Δμ, Δs).
    """

    def __init__(
        self,
        use_tfn_residual: bool = True,
        tfn_scalar_dim: int = 16,
        tfn_vector_dim: int = 4,
    ) -> None:
        super().__init__()
        # Discrete rotation group (cube24) — kept available for utilities
        # such as cross-object transfer.  Not used in the forward path
        # since Encoder.head_h supplies a continuous SO(3) matrix directly.
        self.rot_group = DiscreteRotationGroup()

        self.use_tfn = use_tfn_residual
        if use_tfn_residual:
            self.tfn = EquivariantResidual(tfn_scalar_dim, tfn_vector_dim)

    # ──────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def compose_total_rotation(
        R_h: torch.Tensor,    # [B, K, 3, 3]   coarse / discrete-prior rotation
        xi:  torch.Tensor,    # [B, K, 3]      so(3) micro-rotation
    ) -> torch.Tensor:
        """R_total = exp(ξ) · R_h   →  [B, K, 3, 3]."""
        R_xi = exp_so3(xi)
        return R_xi @ R_h
    
    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        mu:    torch.Tensor,                 # [B, K, N, 3]    Gaussian centres (canonical)
        cov:   torch.Tensor,                 # [B, K, N, 3, 3]
        sh:    torch.Tensor,                 # [B, K, N, C_sh]
        scale: torch.Tensor,                 # [B, K, N, 3]    log-scale
        translation:    torch.Tensor,        # [B, K, 3]       ℓ from physical_params
        rotation:       torch.Tensor,        # [B, K, 3, 3]    R_h from physical_params (already SO(3))
        micro_rotation: torch.Tensor,        # [B, K, 3]       ξ from physical_params
        rho_summary: Optional[torch.Tensor] = None,  # [B, K, 4]  4 PDF physical scalars
                                                     #            (E, ρ_m, μ, damping) — see
                                                     #            DeformSim.physics_summary
        mask: Optional[torch.Tensor] = None,         # [B, K, N] bool  (currently unused; TFN
                                                     #                  outputs on padded are
                                                     #                  filtered downstream)
    ) -> dict:
        """
        Apply rigid SE(3) + optional TFN residual.

        Returns dict with:
            mu     [B, K, N, 3]
            cov    [B, K, N, 3, 3]
            sh     [B, K, N, C_sh]    (unchanged — rotated downstream)
            scale  [B, K, N, 3]       (possibly with TFN log-scale residual)
            R_used [B, K, 3, 3]       composed rotation
            t_used [B, K, 3]          translation
        """
        B, K, N, _ = mu.shape

        # ── Compose total rotation: R = exp(ξ) · R_h ────────────────
        # exp_so3 handles arbitrary leading dims, so no reshape gymnastics.
        R = self.compose_total_rotation(rotation, micro_rotation)      # [B, K, 3, 3]
        R_e = R.unsqueeze(2)                                           # [B, K, 1, 3, 3]
        t   = translation                                              # [B, K, 3]

        # ── μ' = μ R^T + t  (row-vector convention) ─────────────────
        # einsum  "bkni, bkji -> bknj"  ≡  mu @ R.transpose(-1,-2)
        mu_new = torch.einsum("bkni, bkji -> bknj", mu, R) + t.unsqueeze(2)

        # ── Σ' = R Σ R^T ────────────────────────────────────────────
        cov_new = R_e @ cov @ R_e.transpose(-2, -1)

        # ── SH unchanged at the rigid step ──────────────────────────
        # Rotation of l=1 SH band is deferred:
        #   physics ON  → compensate_appearance() rotates SH via R_loc (local)
        #   physics OFF → DeformSim fallback rotates SH via R_rigid (global)
        sh_new = sh

        scale_new = scale.clone()

        # ── Optional TFN equivariant residual ───────────────────────
        if self.use_tfn and rho_summary is not None:
            # Per-Gaussian "radius" feature (norm of log-scale, broadcast-safe)
            gr = scale.norm(dim=-1, keepdim=True).clamp(min=1e-6)      # [B, K, N, 1]
            d_mu, d_s = self.tfn(
                xi=micro_rotation, l=translation,
                rho_summary=rho_summary, gauss_radius=gr,
            )
            # Optional masking: zero residual on padded positions
            if mask is not None:
                m = mask.unsqueeze(-1).to(d_mu.dtype)                  # [B, K, N, 1]
                d_mu = d_mu * m
                d_s  = d_s  * m

            mu_new    = mu_new + d_mu
            scale_new = scale_new + d_s         # additive in log-scale

        return dict(
            mu=mu_new,
            cov=cov_new,
            sh=sh_new,
            scale=scale_new,
            R_used=R,
            t_used=t,
        )
