"""
deform/router.py — Multi-backend physics router with Gumbel-softmax dispatch.

Routes each object to one or more physics backends (rigid_contact / PBD / MPM)
via learned logits.  Two dispatch modes:

  - **soft** (training default): weighted sum of all backend outputs.  Lets
    gradients flow through routing decisions.
  - **hard** (inference / annealed): exclusive winner via straight-through
    Gumbel-softmax.  Used for cleaner inference / discrete routing analysis.

Mask-aware:
  - Forward signature accepts an optional ``mask`` and forwards it to every
    backend's ``simulate(positions, params, mask)``.  Routing logits are
    independent of mask (they depend on ρ only).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..physics import (
    DeformResult,
    DiffDeformer,
    PhysicsParams,
    RigidContactBackend,
    ShapeMatchingPBD,
    MPMBackend,
)
from ...utils import exp_so3, log_so3


class _WrapBackend(nn.Module):
    """Wrap a non-nn.Module DiffDeformer so it can live in nn.ModuleDict.

    Backends that are stateless (no nn.Module subclassing) get this thin
    wrapper so ``nn.ModuleDict`` can register them and PyTorch can track
    devices via ``.to(device)``.
    """
    def __init__(self, backend: DiffDeformer) -> None:
        super().__init__()
        self._backend = backend

    def simulate(self, *args, **kwargs):
        return self._backend.simulate(*args, **kwargs)


class PhysicsRouter(nn.Module):
    """Routes ρ to a weighted combination of physics backends."""

    def __init__(
        self,
        rho_dim: int = 16,
        backends: Optional[Dict[str, DiffDeformer]] = None,
        temperature: float = 1.0,
        hard: bool = False,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        if backends is None:
            backends = {
                "rigid_contact": RigidContactBackend(),
                "pbd":           ShapeMatchingPBD(),
                "mpm":           MPMBackend(),
            }
        self.backend_names = list(backends.keys())

        # Wrap non-nn.Module backends so nn.ModuleDict can hold them
        self.backends = nn.ModuleDict({
            name: (b if isinstance(b, nn.Module) else _WrapBackend(b))
            for name, b in backends.items()
        })
        self.n_backends = len(backends)

        # Routing head: ρ → logits over backends
        _logit = nn.Linear(rho_dim, self.n_backends)
        self.logit_head = nn.utils.spectral_norm(_logit) if use_spectral_norm else _logit
        self.temperature = temperature
        self.hard = hard

    # ──────────────────────────────────────────────────────────────────
    # Routing probabilities
    # ──────────────────────────────────────────────────────────────────

    def route(self, rho: torch.Tensor) -> torch.Tensor:
        """Compute routing probabilities  [M, n_backends]."""
        logits = self.logit_head(rho)                                   # [M, n_backends]
        if self.hard:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=True)
        else:
            return F.softmax(logits / self.temperature, dim=-1)

    # ──────────────────────────────────────────────────────────────────
    # Forward — combine backend outputs by routing weights
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        positions: torch.Tensor,                   # [M, N, 3]
        rho: torch.Tensor,                         # [M, rho_dim]
        params: PhysicsParams,
        mask: Optional[torch.Tensor] = None,       # [M, N] bool — forwarded to backends
    ) -> Tuple[DeformResult, torch.Tensor]:
        """Run all backends, combine outputs by routing weights.

        Returns:
            combined_result : DeformResult (weighted sum across backends)
            route_probs     : [M, n_backends]  for logging / regularisation
        """
        M, N, _ = positions.shape
        dev = positions.device
        dtype = positions.dtype

        probs = self.route(rho)                                         # [M, n_backends]

        # Accumulators (R_loc averaged in so(3) — see below)
        delta_mu_acc  = torch.zeros(M, N, 3,    device=dev, dtype=dtype)
        delta_cov_acc = torch.zeros(M, N, 3, 3, device=dev, dtype=dtype)
        omega_acc     = torch.zeros(M, N, 3,    device=dev, dtype=dtype)
        J_acc         = torch.zeros(M, N,       device=dev, dtype=dtype)

        for i, name in enumerate(self.backend_names):
            w = probs[:, i]                                             # [M]
            res_i = self.backends[name].simulate(positions, params, mask=mask)

            w_e = w[:, None, None]                                      # [M, 1, 1]
            delta_mu_acc  += w_e * res_i.delta_mu
            delta_cov_acc += w_e.unsqueeze(-1) * res_i.delta_cov
            J_acc         += w[:, None] * res_i.J

            # ── R_loc averaged in the Lie algebra so(3) ─────────────
            # Naive  Σ wᵢ Rᵢ  is NOT in SO(3) (no longer orthogonal,
            # det ≠ 1).  Instead we map each Rᵢ to its axis-angle vector
            # ωᵢ = log_so3(Rᵢ), take the convex combination there
            # (which IS valid under linearity of so(3)), and exponentiate
            # back: R_avg = exp_so3(Σ wᵢ ωᵢ).  This always yields a
            # proper rotation, and gradients flow through both heads
            # smoothly.
            omega_i = log_so3(res_i.R_loc)                              # [M, N, 3]
            omega_acc += w_e * omega_i

        R_loc_acc = exp_so3(omega_acc)                                  # [M, N, 3, 3]

        combined = DeformResult(
            delta_mu=delta_mu_acc, delta_cov=delta_cov_acc,
            R_loc=R_loc_acc, J=J_acc,
        )
        return combined, probs
