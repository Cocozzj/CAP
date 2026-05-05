"""
rigid/tfn_residual.py — TFN (Tensor Field Network) equivariant residual.

A small 2-layer scalar+vector equivariant network that produces per-Gaussian
corrections (Δμ, Δs) to compensate for quantisation gaps in the discrete
action token.

Inputs (per object):
    scalar features:   ‖ξ‖, ‖ℓ‖, rho_summary[4], gauss_radius, log(radius) → 8 dims
                        where rho_summary = (E, ρ_m, μ, damping) — 4 PDF-defined
                        physical scalars decoded from ρ via DeformSim.physics_summary
    vector features:   ξ̂, ℓ̂  → 2 unit vectors of dim 3

Outputs (per Gaussian):
    Δμ:   [B, K, N, 3]   position correction (added to rigid output)
    Δs:   [B, K, N, 3]   log-scale correction (additive on log-scale)

Equivariance: outputs transform consistently under rotation of the input
vectors (ξ, ℓ are 3-vectors; the network preserves rotation equivariance
via the radial × direction decomposition).

Lipschitz constraint (PDF §1.4):
    All ``nn.Linear`` layers are wrapped with ``torch.nn.utils.spectral_norm``
    when ``use_spectral_norm=True`` (default).  This bounds the Lipschitz
    constant of the residual to be small, which keeps physics gradients
    well-behaved during end-to-end training.

"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════════
# Helper: optional spectral normalisation
# ═══════════════════════════════════════════════════════════════════════════

def _maybe_sn(layer: nn.Linear, use_spectral_norm: bool) -> nn.Linear:
    """Wrap a Linear layer with spectral_norm to bound its Lipschitz constant.

    PDF §1.4 requires Lipschitz-bounded modules along the physics gradient
    path so that closure / inverse / commutator losses remain well-conditioned
    during end-to-end training.
    """
    return nn.utils.spectral_norm(layer) if use_spectral_norm else layer


# ═══════════════════════════════════════════════════════════════════════════
# §1  Single TFN layer (scalar + vector)
# ═══════════════════════════════════════════════════════════════════════════

class _TFNLayer(nn.Module):
    """One TFN residual block.

    Pipeline:
      s_out = MLP_s([s, ‖v‖])
      r     = MLP_r(s_out)
      g     = sigmoid(MLP_g(s_out))
      v_out = g · r · v             (per-channel gating + radial scaling)

    Equivariance: v is a vector, ‖v‖ is a scalar → the channel-wise scaling
    by (g · r) preserves the direction, so output is rotation-equivariant.
    """

    def __init__(
        self,
        s_dim: int,
        v_dim: int,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.scalar_mlp = nn.Sequential(
            _maybe_sn(nn.Linear(s_dim + v_dim, s_dim), use_spectral_norm),
            nn.GELU(),
            _maybe_sn(nn.Linear(s_dim, s_dim), use_spectral_norm),
        )
        self.radial = nn.Sequential(
            _maybe_sn(nn.Linear(s_dim, v_dim), use_spectral_norm),
            nn.GELU(),
            _maybe_sn(nn.Linear(v_dim, v_dim), use_spectral_norm),
        )
        self.gate = nn.Sequential(
            _maybe_sn(nn.Linear(s_dim, v_dim), use_spectral_norm),
            nn.Sigmoid(),
        )

    def forward(
        self,
        s: torch.Tensor,        # [..., s_dim]
        v: torch.Tensor,        # [..., v_dim, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_n = v.norm(dim=-1)                                  # [..., v_dim]
        s_out = self.scalar_mlp(torch.cat([s, v_n], -1))      # [..., s_dim]
        rw    = self.radial(s_out).unsqueeze(-1)              # [..., v_dim, 1]
        g     = self.gate(s_out).unsqueeze(-1)                # [..., v_dim, 1]
        v_out = g * rw * v                                     # [..., v_dim, 3]
        return s_out, v_out


# ═══════════════════════════════════════════════════════════════════════════
# §2  EquivariantResidual — public 2-layer TFN
# ═══════════════════════════════════════════════════════════════════════════

class EquivariantResidual(nn.Module):
    """Two-layer TFN that outputs per-Gaussian (Δμ, Δs) corrections.

    Per-Gaussian input is built from:
      scalar:  ‖ξ‖, ‖ℓ‖, rho_summary[4], gauss_radius, log(radius)  → 8 dims
      vector:  ξ̂, ℓ̂  → 2 unit vectors of dim 3

    All Linear layers (input projections + 2 TFN blocks + 2 output heads)
    are wrapped with ``spectral_norm`` by default to satisfy PDF §1.4's
    Lipschitz constraint.
    """

    def __init__(
        self,
        s_dim: int = 16,
        v_dim: int = 4,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        # Input projections
        self.s_in = _maybe_sn(nn.Linear(8, s_dim), use_spectral_norm)
        self.v_in = _maybe_sn(nn.Linear(2, v_dim), use_spectral_norm)

        # Two TFN equivariant blocks
        self.l1 = _TFNLayer(s_dim, v_dim, use_spectral_norm=use_spectral_norm)
        self.l2 = _TFNLayer(s_dim, v_dim, use_spectral_norm=use_spectral_norm)

        # Per-Gaussian output heads
        self.head_mu = _maybe_sn(nn.Linear(v_dim, 1), use_spectral_norm)
        self.head_s  = _maybe_sn(nn.Linear(s_dim, 3), use_spectral_norm)

    def forward(
        self,
        xi: torch.Tensor,              # [B, K, 3]      micro-rotation residual
        l:  torch.Tensor,              # [B, K, 3]      translation
        rho_summary: torch.Tensor,     # [B, K, 4]      (E, ρ_m, μ, damping) PDF scalars
        gauss_radius: torch.Tensor,    # [B, K, N, 1]   per-Gaussian radius
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            delta_mu: [B, K, N, 3]
            delta_s:  [B, K, N, 3]   log-scale increment
        """
        B, K, N, _ = gauss_radius.shape

        # ── Build scalar features  [B, K, N, 8] ──────────────────────
        xi_n  = xi.norm(-1, keepdim=True).unsqueeze(2).expand(B, K, N, 1)
        l_n   = l.norm(-1, keepdim=True).unsqueeze(2).expand(B, K, N, 1)
        rho_e = rho_summary.unsqueeze(2).expand(B, K, N, 4)
        s = torch.cat([
            xi_n, l_n, rho_e,
            gauss_radius,
            gauss_radius.log().clamp(-6, 6),
        ], -1)

        # ── Build vector features  [B, K, N, 2, 3] ───────────────────
        xi_d = F.normalize(xi, dim=-1).unsqueeze(2).expand(B, K, N, 3)
        l_d  = F.normalize(l + 1e-8, dim=-1).unsqueeze(2).expand(B, K, N, 3)
        v = torch.stack([xi_d, l_d], dim=-2)                  # [B, K, N, 2, 3]

        # ── Two TFN layers ───────────────────────────────────────────
        s = self.s_in(s)
        v = self.v_in(v.transpose(-2, -1)).transpose(-2, -1)  # [B, K, N, v_dim, 3]
        s, v = self.l1(s, v)
        s, v = self.l2(s, v)

        # ── Per-Gaussian heads ───────────────────────────────────────
        delta_mu = self.head_mu(v.transpose(-2, -1)).squeeze(-1)   # [B, K, N, 3]
        delta_s  = self.head_s(s)                                   # [B, K, N, 3]
        return delta_mu, delta_s
