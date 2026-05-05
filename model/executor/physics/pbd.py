"""
physics/pbd.py — Position-Based Dynamics via SVD shape matching.

Steps:
  1. Compute optimal rotation R* from current to rest pose via SVD on H = q^T p
     (Mueller et al., "Meshless Deformations Based on Shape Matching", 2005)
  2. Project each particle towards its rotated rest position
  3. Apply iterative damping (PBD style)

Mask-aware:
  - COM (current and rest) computed via masked_mean
  - SVD aggregation H = q^T (p · mask) → padded particles do NOT bias R*
  - All N positions get displaced (padded ones harmlessly), but downstream
    aggregations use the same mask so padded deltas are filtered out

Rest-pose handling:
  - Stateless by default (each call uses input as rest)
  - Optional EMA tracking via ``set_rest_pose()`` + ``use_ema_rest=True``
  - Always call ``reset_rest_cache()`` between scenes to avoid stale cache

Output:
  - delta_mu  : per-particle displacement towards rest goal
  - delta_cov : 0 (PBD only updates positions, not Σ shape)
  - R_loc     : R* broadcast to all particles
  - J         : product of SVD singular values (for opacity/scale compensation)
"""

from __future__ import annotations

import warnings
from typing import Optional

import torch

from .base import DiffDeformer, DeformResult, PhysicsParams
from ...utils import masked_mean


class ShapeMatchingPBD(DiffDeformer):
    """Shape-matching PBD with optional EMA rest-pose tracking.

    .. warning::

        When ``use_ema_rest=True`` (default is False), this backend caches a
        running EMA of the "rest" pose across forward calls.  This cache is
        **module-level state** that PERSISTS across scenes — neither
        ``DeformSim`` nor ``Executor`` resets it automatically.

        If you enable EMA rest, you MUST call ``reset_rest_cache()`` (or
        ``set_rest_pose(new_rest)``) at every scene / episode boundary.
        Forgetting this lets the previous scene's rest pose contaminate the
        next scene's dynamics in a way that is silent and very hard to debug.

        The default ``use_ema_rest=False`` path is stateless and safe.
    """

    def __init__(
        self,
        stiffness: float = 0.8,
        damping: float = 0.05,
        ema_decay: float = 0.99,
        use_ema_rest: bool = False,
    ) -> None:
        self.stiffness = stiffness
        self.base_damping = damping
        self.ema_decay = ema_decay
        self.use_ema_rest = use_ema_rest
        self._rest_cache: Optional[torch.Tensor] = None

        if use_ema_rest:
            warnings.warn(
                "ShapeMatchingPBD(use_ema_rest=True) keeps a stateful rest-pose "
                "cache across forward calls.  Call reset_rest_cache() between "
                "scenes/episodes to avoid silent contamination.",
                stacklevel=2,
            )

    def reset_rest_cache(self) -> None:
        """Clear rest-pose cache.  MUST be called between scenes when
        ``use_ema_rest=True``."""
        self._rest_cache = None

    def set_rest_pose(self, positions: torch.Tensor) -> None:
        """Explicitly set rest pose for a new sequence (replaces cache)."""
        self._rest_cache = positions.detach().clone()

    def simulate(
        self,
        positions: torch.Tensor,                   # [M, N, 3]
        params: PhysicsParams,
        mask: Optional[torch.Tensor] = None,       # [M, N] bool
    ) -> DeformResult:
        M, N, _ = positions.shape
        dev = positions.device
        dtype = positions.dtype

        # ── Choose rest pose ─────────────────────────────────────────
        if (self.use_ema_rest
                and self._rest_cache is not None
                and self._rest_cache.shape == positions.shape
                and self._rest_cache.device == positions.device):
            rest = self._rest_cache
        else:
            rest = positions.detach()
            if self.use_ema_rest:
                self._rest_cache = rest.clone()

        # ── Mask-aware centroids ─────────────────────────────────────
        if mask is not None:
            cm_cur  = masked_mean(positions, mask, dim=1, keepdim=True)   # [M, 1, 3]
            cm_rest = masked_mean(rest,      mask, dim=1, keepdim=True)
            mask_f  = mask.float().unsqueeze(-1)                          # [M, N, 1]
        else:
            cm_cur  = positions.mean(1, keepdim=True)
            cm_rest = rest.mean(1, keepdim=True)
            mask_f  = None

        q = positions - cm_cur                                            # [M, N, 3]  centred current
        p = rest - cm_rest                                                # [M, N, 3]  centred rest

        # ── Apply external force as displacement prior ───────────────
        f_disp = params.ext_force.unsqueeze(1) * params.dt.unsqueeze(-1)  # [M, 1, 3]
        q = q + f_disp.expand(M, N, 3)

        # ── Mask-aware SVD shape matching ────────────────────────────
        # H = sum over real particles of q_i p_i^T
        if mask_f is not None:
            H_mat = (q * mask_f).transpose(1, 2) @ (p * mask_f)           # [M, 3, 3]
        else:
            H_mat = q.transpose(1, 2) @ p

        # Add small ridge for numerical stability (avoid degenerate gradient)
        H_mat = H_mat + 1e-6 * torch.eye(3, device=dev, dtype=dtype).unsqueeze(0)

        # SVD (cast to float32 under AMP for stability)
        H_f32 = H_mat.float() if H_mat.dtype != torch.float32 else H_mat
        U, S, Vt = torch.linalg.svd(H_f32)

        # Ensure proper rotation (det = +1) via sign flip on last singular vector
        det = torch.det(U @ Vt)
        sign = torch.ones(M, 3, device=dev, dtype=H_f32.dtype)
        sign[:, -1] = det.sign()
        R_opt = (U * sign.unsqueeze(-2) @ Vt).to(dtype)                   # [M, 3, 3]

        # ── Goal positions: R* @ p + cm_cur ──────────────────────────
        goal = (R_opt[:, None] @ p.unsqueeze(-1)).squeeze(-1) + cm_cur    # [M, N, 3]

        # ── Iterative projection with damping ────────────────────────
        damp = (self.base_damping + params.damping.unsqueeze(-1)).clamp(0, 1)  # [M, 1, 1]
        x = positions.clone()
        for _ in range(max(params.n_iters, 1)):
            x = x + self.stiffness * (goal - x) * (1 - damp)

        delta_mu = x - positions

        # ── Local rotation per Gaussian ≈ global R* (shape-matching) ─
        R_loc = R_opt[:, None].expand(M, N, 3, 3)

        # ── Volume Jacobian from SVD singular values ─────────────────
        # det(R) = ±1, |J| = prod(S); clamp to keep opacity/scale finite
        J = S.to(dtype).prod(-1, keepdim=True).expand(M, N).clamp(min=0.1)  # [M, N]

        # PBD doesn't update cov directly (only positions move)
        delta_cov = torch.zeros(M, N, 3, 3, device=dev, dtype=dtype)

        # ── Optional EMA update of rest pose for next step ───────────
        if self.use_ema_rest and self._rest_cache is not None:
            self._rest_cache = (
                self.ema_decay * self._rest_cache
                + (1 - self.ema_decay) * x.detach()
            )

        return DeformResult(delta_mu=delta_mu, delta_cov=delta_cov,
                            R_loc=R_loc, J=J)
