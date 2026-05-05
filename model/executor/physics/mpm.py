"""
physics/mpm.py — Material Point Method (DiffTaichi bridge).

Lazy-loads the optional ``diff_taichi_bridge`` module.  When unavailable,
returns zero deformation so the PhysicsRouter naturally down-weights this
backend via Gumbel routing (i.e. the model learns to ignore MPM if the
bridge isn't installed).

Gradient flow is preserved through ``_MPMBridgeFunction`` — a custom
autograd Function that uses DiffTaichi's native ``grad_mpm`` if available,
or falls back to per-axis finite-difference Jacobian estimation.

Mask-aware behaviour:
  Currently the MPM bridge treats every input particle as real (no mask
  support in DiffTaichi).  We rely on the upstream caller to either:
    (a) extract real particles before calling, or
    (b) accept that padded particles at the origin contribute "ghost"
        particles to the simulation.  For typical PartNet-Mobility data
        with low padding waste this is acceptable.
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from .base import DiffDeformer, DeformResult, PhysicsParams


# Module-level guard so we only warn ONCE per process about FD fallback,
# instead of spamming on every backward pass during training.
_FD_WARN_EMITTED = False


# ═══════════════════════════════════════════════════════════════════════════
# §1  Custom autograd Function for DiffTaichi bridge
# ═══════════════════════════════════════════════════════════════════════════

class _MPMBridgeFunction(torch.autograd.Function):
    """Custom autograd wrapper around an external DiffTaichi MPM step.

    Forward:  Calls ``bridge.step_mpm`` with detached inputs.
    Backward: Uses ``bridge.grad_mpm`` if available; otherwise a 3-axis
              finite-difference Jacobian-vector product on positions.
    """

    @staticmethod
    def forward(ctx, positions, bridge, params):
        """
        Args:
            positions: [M, N, 3] Gaussian centres (requires_grad=True in graph)
            bridge:    loaded diff_taichi_bridge module
            params:    PhysicsParams dataclass
        Returns:
            new_positions: [M, N, 3]
            R_loc:         [M, N, 3, 3]
            J:             [M, N]
        """
        M, N, _ = positions.shape
        dev = positions.device

        result = bridge.step_mpm(
            positions=positions.detach(),
            youngs=params.youngs_modulus.squeeze(-1).detach(),
            poisson=params.poisson_ratio.squeeze(-1).detach(),
            density=params.density.squeeze(-1).detach(),
            ext_force=params.ext_force.detach(),
            dt=params.dt.squeeze(-1).detach(),
            n_steps=params.n_substeps,
        )

        new_pos = result["positions"].to(dev)
        R_loc = result.get("R_loc",
                           torch.eye(3, device=dev, dtype=positions.dtype)
                                .expand(M, N, 3, 3))
        J = result.get("J", torch.ones(M, N, device=dev, dtype=positions.dtype))

        # Save context for backward
        ctx.save_for_backward(positions, new_pos)
        ctx.bridge = bridge
        ctx.params = params
        ctx.fd_eps = 1e-4

        return new_pos, R_loc, J

    @staticmethod
    def backward(ctx, grad_new_pos, grad_R_loc, grad_J):
        """Use native grad_mpm if available; otherwise FD Jacobian on positions."""
        positions, new_pos_fwd = ctx.saved_tensors
        bridge = ctx.bridge
        params = ctx.params
        eps = ctx.fd_eps
        dev = positions.device

        if hasattr(bridge, "grad_mpm"):
            grad_pos = bridge.grad_mpm(
                positions=positions.detach(),
                grad_output=grad_new_pos.detach(),
                youngs=params.youngs_modulus.squeeze(-1).detach(),
                poisson=params.poisson_ratio.squeeze(-1).detach(),
                density=params.density.squeeze(-1).detach(),
                ext_force=params.ext_force.detach(),
                dt=params.dt.squeeze(-1).detach(),
                n_steps=params.n_substeps,
            ).to(dev)
            return grad_pos, None, None

        # ── Finite-difference Jacobian-vector product (4× cost) ──────
        # Approximate (∂new_pos/∂pos · grad_new_pos) per spatial axis.
        # Warn ONCE per process so users notice the perf hit and install
        # a DiffTaichi build that exposes ``grad_mpm`` if they care.
        global _FD_WARN_EMITTED
        if not _FD_WARN_EMITTED:
            warnings.warn(
                "MPMBackend: bridge does not expose grad_mpm — falling back to "
                "3-axis finite-difference Jacobian.  This runs MPM 4× per backward "
                "step and will dominate training time.  Install a DiffTaichi build "
                "with native grad_mpm to recover speed.",
                RuntimeWarning, stacklevel=2,
            )
            _FD_WARN_EMITTED = True

        grad_pos = torch.zeros_like(positions)
        pos_det = positions.detach()
        for d in range(3):
            perturb = torch.zeros_like(positions)
            perturb[..., d] = eps
            result_plus = bridge.step_mpm(
                positions=pos_det + perturb,
                youngs=params.youngs_modulus.squeeze(-1).detach(),
                poisson=params.poisson_ratio.squeeze(-1).detach(),
                density=params.density.squeeze(-1).detach(),
                ext_force=params.ext_force.detach(),
                dt=params.dt.squeeze(-1).detach(),
                n_steps=params.n_substeps,
            )
            delta = (result_plus["positions"].to(dev) - new_pos_fwd) / eps
            grad_pos[..., d] = (delta * grad_new_pos).sum(dim=-1)

        return grad_pos, None, None


# ═══════════════════════════════════════════════════════════════════════════
# §2  MPMBackend — public interface
# ═══════════════════════════════════════════════════════════════════════════

class MPMBackend(DiffDeformer):
    """Material Point Method via DiffTaichi (optional)."""

    def __init__(self) -> None:
        self._bridge = None
        self._available: Optional[bool] = None

    def _try_load(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import importlib
            self._bridge = importlib.import_module("diff_taichi_bridge")
            self._available = True
        except ImportError:
            self._available = False
        return self._available

    def simulate(
        self,
        positions: torch.Tensor,                   # [M, N, 3]
        params: PhysicsParams,
        mask: Optional[torch.Tensor] = None,       # currently unused (MPM treats all)
    ) -> DeformResult:
        M, N, _ = positions.shape
        dev = positions.device
        dtype = positions.dtype

        if self._try_load() and self._bridge is not None:
            # Custom autograd Function preserves gradient flow through DiffTaichi
            new_pos, R_loc, J = _MPMBridgeFunction.apply(
                positions, self._bridge, params,
            )
            delta_mu = new_pos - positions
        else:
            # Graceful fallback: zero deformation
            delta_mu = torch.zeros(M, N, 3, device=dev, dtype=dtype)
            R_loc    = torch.eye(3, device=dev, dtype=dtype).expand(M, N, 3, 3)
            J        = torch.ones(M, N, device=dev, dtype=dtype)

        delta_cov = torch.zeros(M, N, 3, 3, device=dev, dtype=dtype)
        return DeformResult(delta_mu=delta_mu, delta_cov=delta_cov,
                            R_loc=R_loc, J=J)
