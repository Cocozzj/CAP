"""
deform/sim.py — DeformSim: orchestrates non-rigid deformation.

Pipeline (when ``enable_physics=True``):
    ρ → RhoParser → PhysicsParams
                  ↓
         VelocityFieldTFN  →  Δμ_v  (Verlet integration)
                  ↓
       PhysicsRouter  →  (Δμ_p, Δcov, R_loc, J)  (mixture of backends)
                  ↓
       Combine: μ_new = μ + Δμ_v + Δμ_p
                cov_new = update via log-Euclidean (or linear)
                  ↓
       Appearance compensation: rotate SH, scale opacity by 1/J, scale by J^(1/3)

When ``enable_physics=False``:
    BlackBoxFallback only — Δscale + Δopacity from ρ; no position change.
    SH is rotated by the rigid R passed in (since the rigid branch deferred SH
    rotation when physics is involved).

Mask-aware:
    - All physics calls receive the [B, K, N] mask, flattened to [M, N]
    - Appearance compensation is per-element (no aggregation), no mask needed
    - Combined displacement is clamped + nan-guarded for safety

"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..physics import (
    PhysicsParams,
    RigidContactBackend,
    ShapeMatchingPBD,
    MPMBackend,
)
from ...utils import (
    cov_to_log_euclidean,
    log_euclidean_to_cov,
    project_spd,
)

from .rho_parser import RhoParser
from .velocity_field import VelocityFieldTFN
from .router import PhysicsRouter
from .fallback import BlackBoxFallback

# ═══════════════════════════════════════════════════════════════════════════
# §1  SH (l=1) rotation
# ═══════════════════════════════════════════════════════════════════════════

def rotate_sh_l1(sh: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Rotate the l=1 SH band (channels 1..4) by rotation matrix R.

    The l=0 band (DC term) is rotation-invariant.  The l=1 band transforms
    like a 3-vector under SO(3):  sh_l1' = R · sh_l1.  Higher bands (l ≥ 2)
    are approximated as identity here — exact rotation requires Wigner-D
    matrices and is rarely needed for the small per-token rotations in CAP.

    Args:
        sh:  [..., N, C_sh]    SH coefficients (assumes layout
                                [DC, l1_x, l1_y, l1_z, l2_..., ...])
        R:   [..., N, 3, 3]    per-particle rotation (broadcast-compatible)

    Returns:
        sh':  [..., N, C_sh]   with l=1 band rotated, other bands unchanged
    """
    if sh.shape[-1] < 4:
        # No l=1 band present (e.g., DC-only colour) → nothing to do
        return sh

    out = sh.clone()
    # Rotate l=1 band: matrix-vector product per particle
    out[..., 1:4] = (R @ sh[..., 1:4].unsqueeze(-1)).squeeze(-1)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# §2  Volume Jacobian compensation (opacity + scale)
# ═══════════════════════════════════════════════════════════════════════════

def compensate_appearance(
    sh: torch.Tensor,            # [..., N, C_sh]
    opacity: torch.Tensor,       # [..., N, 1]
    scale: torch.Tensor,         # [..., N, 3]
    R_loc: torch.Tensor,         # [..., N, 3, 3]
    J: torch.Tensor,             # [..., N]
    j_floor: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply per-particle local-rotation SH compensation and volume-Jacobian
    opacity/scale compensation in one shot.

    Steps:
      1. SH l=1 ← R_loc · SH l=1
      2. opacity ← opacity / J        (clamp J to avoid division by ~0)
      3. scale   ← scale * J^(1/3)    (uniform per-axis isotropic compensation)

    Args:
        sh:       [..., N, C_sh]
        opacity:  [..., N, 1]
        scale:    [..., N, 3]      log-scale (we treat the J factor in linear
                                   space and add log(J^(1/3)) implicitly via
                                   multiplication on linear scale; the actual
                                   semantics depend on caller's convention)
        R_loc:    [..., N, 3, 3]   local rotation per particle
        J:        [..., N]         volume Jacobian per particle
        j_floor:  lower bound on J for division stability

    Returns:
        (sh', opacity', scale')
    """
    sh_out = rotate_sh_l1(sh, R_loc)

    J_c = J.clamp(min=j_floor).unsqueeze(-1)                 # [..., N, 1]
    opacity_out = opacity * (1.0 / J_c)
    scale_out   = scale * J_c.pow(1.0 / 3.0)
    return sh_out, opacity_out, scale_out


class DeformSim(nn.Module):
    """Unified non-rigid deformation module.

    Parameters
    ----------
    rho_dim : int
        Dimension of ρ vector from Encoder.physical_params["deformation"].
    task_dim : int
        Dimension of task context (0 = no task conditioning).
    param_mode : str
        Covariance update domain: ``"logeuclid"`` (recommended) or ``"linear"``.
    """

    # Available physics backend names (used by enabled_backends ablation).
    _ALL_BACKENDS = ("rigid_contact", "pbd", "mpm")

    def __init__(
        self,
        rho_dim: int = 16,
        task_dim: int = 0,
        param_mode: str = "logeuclid",
        router_temperature: float = 1.0,
        router_hard: bool = False,
        max_delta_mu: float = 1.0,
        # ── Optional config passthroughs (each None → component default) ──
        # PBD's n_iters / n_substeps come in via rho_parser_cfg (RhoParser owns
        # those hyperparameters and threads them into PhysicsParams), NOT via
        # a separate pbd_backend_cfg here.  ShapeMatchingPBD currently has no
        # constructor kwargs the model would override.
        rho_parser_cfg: Optional[Dict[str, Any]] = None,
        rigid_contact_cfg: Optional[Dict[str, Any]] = None,
        # ── Ablation: restrict to a subset of physics backends ──
        enabled_backends: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            max_delta_mu: per-step displacement clamp (metres).  Combined
                          (vfield + physics) Δμ is capped to ±max_delta_mu
                          per axis as a NaN/explosion safety net.  Tune to
                          your scene scale (~0.1 for tabletop, ~10 for room).
            rho_parser_cfg:    extra kwargs for RhoParser (e.g.
                               ``{"n_iters": 5, "n_substeps": 2}``).
            rigid_contact_cfg: kwargs for RigidContactBackend.
            enabled_backends:  list of backend names to instantiate (default =
                               all 3).  Use to ablate single-backend variants
                               (PDF main proposal §5.2 ablation table — e.g.
                               ``["pbd"]`` for PBD-only, ``["pbd", "mpm"]``
                               for soft-only).  Routing logits adapt to the
                               smaller backend count automatically.
        """
        super().__init__()
        self.param_mode  = param_mode
        self.max_delta_mu = float(max_delta_mu)

        self.rho_parser = RhoParser(rho_dim, task_dim, **(rho_parser_cfg or {}))
        self.vfield     = VelocityFieldTFN(rho_dim)

        # Validate / default the enabled-backend set
        enabled = list(enabled_backends) if enabled_backends else list(self._ALL_BACKENDS)
        invalid = [b for b in enabled if b not in self._ALL_BACKENDS]
        if invalid:
            raise ValueError(
                f"Unknown physics backend(s): {invalid}. "
                f"Valid choices: {list(self._ALL_BACKENDS)}"
            )
        if not enabled:
            raise ValueError("At least one physics backend must be enabled.")

        # Build only the requested backends; PhysicsRouter's logit head sizes to len(backends)
        backend_factories = {
            "rigid_contact": lambda: RigidContactBackend(**(rigid_contact_cfg or {})),
            "pbd":           lambda: ShapeMatchingPBD(),
            "mpm":           lambda: MPMBackend(),
        }
        backends = {name: backend_factories[name]() for name in enabled}

        self.router     = PhysicsRouter(
            rho_dim,
            backends=backends,
            temperature=router_temperature,
            hard=router_hard,
        )
        self.fallback   = BlackBoxFallback(rho_dim)

        # Optional inference-time PhysicsParams overrides for ablation /
        # generalisation experiments.  Empty dict = no override (default).
        # See `set_param_override` docstring for the full list of supported
        # keys and the corresponding PDF references.
        self._param_overrides: Dict[str, Union[float, torch.Tensor]] = {}

    # ──────────────────────────────────────────────────────────────────
    # Physics-parameter override hooks (ablation / generalisation)
    # ──────────────────────────────────────────────────────────────────
    #
    # Supported keys (must match PhysicsParams field names):
    #     youngs_modulus   scalar  — Young's modulus E             [M, 1]
    #     poisson_ratio    scalar  — Poisson ratio  ν              [M, 1]
    #     density          scalar  — mass density   ρ_m  (kg/m³)   [M, 1]
    #     ext_force        3-vec   — external force vector         [M, 3]
    #     friction_coeff   scalar  — Coulomb friction μ            [M, 1]
    #     damping          scalar  — velocity damping              [M, 1]
    #     dt               scalar  — per-object timestep           [M, 1]
    #     n_iters          int     — PBD iteration count
    #     n_substeps       int     — Verlet substep count
    #
    # PDF references:
    #   f07d2c0a (物理插件方案) 实验表        — 跨材料 / 跨质量 / 跨摩擦泛化
    #   fdfa011c (主方案) 行 1101            — "反事实验证：改变物理条件验证
    #                                          输出（如仅改变质量参数）"
    #
    # Examples
    # --------
    #     # 行 1101 反事实质量替换
    #     deform.set_param_override(density=7800.)              # 铁
    #     ... inference ...
    #     deform.clear_param_override()
    #
    #     # 跨材料 + 跨摩擦
    #     deform.set_param_override(youngs_modulus=1e6, friction_coeff=0.3)

    def set_param_override(
        self, **overrides: Union[float, torch.Tensor]
    ) -> None:
        """Override one or more learned PhysicsParams fields at inference.

        Each call REPLACES the previous override set (use `clear_param_override`
        explicitly to reset to learned values).

        Args:
            **overrides:  field_name=value  pairs.  Scalar floats are
                          broadcast to per-object tensors at forward time;
                          tensors are reshaped/cast to the learned field's
                          shape & dtype.

        Raises:
            ValueError: if a key is not a valid PhysicsParams field name.
        """
        valid = {f.name for f in dataclasses.fields(PhysicsParams)}
        invalid = set(overrides.keys()) - valid
        if invalid:
            raise ValueError(
                f"Unknown PhysicsParams field(s): {sorted(invalid)}. "
                f"Valid keys: {sorted(valid)}"
            )
        self._param_overrides = dict(overrides)
        warnings.warn(
            f"DeformSim.set_param_override active: {self._param_overrides}. "
            "Call clear_param_override() before resuming training to avoid "
            "silently using overridden values.",
            stacklevel=2,
        )

    def clear_param_override(self) -> None:
        """Remove all parameter overrides, reverting to learned values."""
        self._param_overrides = {}

    # ── Backward-compat shims (density-only API) ──────────────────────

    def set_density_override(self, density: float) -> None:
        """Deprecated. Use ``set_param_override(density=...)``."""
        warnings.warn(
            "set_density_override is deprecated; "
            "use set_param_override(density=...) instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.set_param_override(density=density)

    def clear_density_override(self) -> None:
        """Deprecated. Use ``clear_param_override()``."""
        warnings.warn(
            "clear_density_override is deprecated; "
            "use clear_param_override() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self.clear_param_override()

    # ──────────────────────────────────────────────────────────────────
    # Public ρ-decoding helpers (called by upstream Executor)
    # ──────────────────────────────────────────────────────────────────

    def physics_summary(
        self,
        rho_flat: torch.Tensor,                     # [M, rho_dim]   M = B·K
        task_context_flat: Optional[torch.Tensor] = None,   # [M, task_dim]
    ) -> Tuple[torch.Tensor, PhysicsParams]:
        """Decode ρ → (rho_summary, full PhysicsParams) for upstream conditioning.

        ``rho_summary`` is a 4-tensor of (E, ρ_m, μ, damping) used by the
        rigid branch's TFN residual to condition on physical scalars (PDF
        f07d2c0a §1.3 / fdfa011c 行 197 — execution-time physical quantities).

        Calling this and then passing the returned ``params`` back into
        ``forward(precomputed_params=...)`` lets the caller AVOID running
        ``rho_parser`` twice on the same ρ — important if RhoParser ever
        gains stochastic layers (dropout, noise injection), where a second
        decode would yield DIFFERENT params and silently desync the rigid
        and deform branches' physics conditioning.

        Args:
            rho_flat:           [M, rho_dim]
            task_context_flat:  [M, task_dim]  optional task conditioning

        Returns:
            rho_summary: [M, 4]   concatenated (E, ρ_m, μ, damping)
            params:      PhysicsParams  full decoded params (incl. ext_force,
                         poisson_ratio, dt, n_iters, n_substeps), suitable to
                         pass back into ``forward(precomputed_params=...)``.
        """
        params = self.rho_parser(rho_flat, task_context_flat)
        summary = torch.cat([
            params.youngs_modulus,    # E
            params.density,           # ρ_m
            params.friction_coeff,    # μ
            params.damping,           # damping
        ], dim=-1)                    # [M, 4]
        return summary, params

    def _apply_param_overrides(self, params: PhysicsParams) -> PhysicsParams:
        """Apply self._param_overrides on top of decoded PhysicsParams.

        Tensors are cast/broadcast to match the learned field's shape & dtype;
        scalar ints (n_iters, n_substeps) are coerced via int(...).
        """
        if not self._param_overrides:
            return params
        new_fields: Dict[str, Any] = {}
        for k, v in self._param_overrides.items():
            old = getattr(params, k)
            if isinstance(old, torch.Tensor):
                if isinstance(v, torch.Tensor):
                    new_fields[k] = v.to(device=old.device, dtype=old.dtype) \
                                     .expand_as(old)
                else:
                    new_fields[k] = torch.full_like(old, float(v))
            else:
                # int scalar field (n_iters, n_substeps)
                new_fields[k] = type(old)(v)
        return dataclasses.replace(params, **new_fields)

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        mu:      torch.Tensor,                  # [B, K, N, 3]
        cov:     torch.Tensor,                  # [B, K, N, 3, 3]
        sh:      torch.Tensor,                  # [B, K, N, C_sh]
        opacity: torch.Tensor,                  # [B, K, N, 1]
        scale:   torch.Tensor,                  # [B, K, N, 3]
        rho:     torch.Tensor,                  # [B, K, rho_dim]
        enable_physics: bool = True,
        task_context:   Optional[torch.Tensor] = None,   # [B, K, task_dim]
        R_rigid:        Optional[torch.Tensor] = None,   # [B, K, 3, 3]
        mask:           Optional[torch.Tensor] = None,   # [B, K, N] bool
        precomputed_params: Optional[PhysicsParams] = None,
    ) -> dict:
        """Apply non-rigid deformation (or fallback if physics disabled).

        Args:
            ...
            precomputed_params: optional pre-decoded PhysicsParams (e.g. from a
                prior ``physics_summary`` call by the Executor).  When given,
                ``rho_parser`` is NOT called again — this avoids double-decoding
                the same ρ across the rigid and deform branches.  Ignored on
                the physics-OFF path (which uses BlackBoxFallback instead of
                RhoParser).

        Returns dict:
          mu, cov, sh, opacity, scale   — updated Gaussian params
          route_probs                    — [B, K, n_backends] (None if fallback)
          vfield_delta                   — [B, K, N, 3]      velocity-field displacement
        """
        B, K, N, _ = mu.shape
        dev = mu.device
        dtype = mu.dtype

        # ════════════════════════════════════════════════════════════
        # Path A: physics disabled — black-box appearance update only
        # ════════════════════════════════════════════════════════════
        if not enable_physics:
            ds, do = self.fallback(rho)                                  # [B,K,3], [B,K,1]

            # SH rotation by rigid R (rigid branch deferred this when physics is
            # involved; here we apply it ourselves since there's no R_loc).
            sh_out = sh
            if R_rigid is not None:
                R_exp = R_rigid[:, :, None, :, :].expand(B, K, N, 3, 3)
                sh_flat   = sh.reshape(B * K * N, -1)
                R_flat    = R_exp.reshape(B * K * N, 3, 3)
                sh_out = rotate_sh_l1(sh_flat, R_flat).reshape(B, K, N, -1)

            return dict(
                mu=mu,
                cov=cov,
                sh=sh_out,
                opacity=opacity + do.unsqueeze(2).expand(B, K, N, 1),
                scale=scale + ds.unsqueeze(2).expand(B, K, N, 3),
                route_probs=None,
                vfield_delta=torch.zeros(B, K, N, 3, device=dev, dtype=dtype),
            )

        # ════════════════════════════════════════════════════════════
        # Path B: physics enabled — full pipeline
        # ════════════════════════════════════════════════════════════

        # ── Flatten leading (B, K) dims to M = B·K for backends ──────
        rho_flat  = rho.reshape(B * K, -1)
        tc_flat   = task_context.reshape(B * K, -1) if task_context is not None else None
        mu_flat   = mu.reshape(B * K, N, 3)
        mask_flat = mask.reshape(B * K, N) if mask is not None else None

        # ── Decode ρ → PhysicsParams (or reuse caller's pre-decoded copy) ─
        # Reusing precomputed_params makes the rigid + deform branches share
        # ONE rho_parser pass per token, which is both faster and required
        # for correctness if rho_parser ever gains stochastic layers.
        if precomputed_params is not None:
            params = precomputed_params
        else:
            params = self.rho_parser(rho_flat, tc_flat)

        # Optional inference-time overrides for PDF-required ablations
        # (跨材料/跨质量/跨摩擦 + 行 1101 反事实).  No-op when empty.
        params = self._apply_param_overrides(params)

        # ── Velocity field (mask-aware COM) ──────────────────────────
        vf_delta = self.vfield(
            mu_flat, rho_flat, params.dt,
            n_substeps=params.n_substeps, mask=mask_flat,
        )                                                                # [M, N, 3]
        # Safety: TFN-based vfield can produce inf/NaN on first physics-enabled
        # step (untrained network seeing in-distribution ρ for the first time).
        # Clamp here so the value that gets stored in aux["vfield_delta"]
        # (line below at the dict return) is always finite — otherwise
        # physics_loss.vol = ‖vfield_delta‖² → NaN, poisoning every loss
        # term that reads aux (closure / inverse / commutator / physics_*).
        vf_delta = torch.nan_to_num(vf_delta, nan=0.0, posinf=0.0, neginf=0.0)
        vf_delta = vf_delta.clamp(-self.max_delta_mu, self.max_delta_mu)

        # ── Physics backends (mask-aware) ────────────────────────────
        mu_for_phys = mu_flat + vf_delta
        phys_result, route_probs = self.router(
            mu_for_phys, rho_flat, params, mask=mask_flat,
        )

        # ── Combine displacements + safety guard ─────────────────────
        total_delta_mu = vf_delta + phys_result.delta_mu
        total_delta_mu = torch.nan_to_num(total_delta_mu, nan=0.0, posinf=0.0, neginf=0.0)
        total_delta_mu = total_delta_mu.clamp(-self.max_delta_mu, self.max_delta_mu)

        mu_new = mu + total_delta_mu.reshape(B, K, N, 3)

        # ── Covariance update ────────────────────────────────────────
        if self.param_mode == "logeuclid":
            logS     = cov_to_log_euclidean(cov)
            logS_new = logS + phys_result.delta_cov.reshape(B, K, N, 3, 3)
            cov_new  = log_euclidean_to_cov(logS_new)
            cov_new  = project_spd(cov_new)
        else:
            cov_new = cov + phys_result.delta_cov.reshape(B, K, N, 3, 3)
            cov_new = project_spd(cov_new)

        # ── Appearance compensation (SH rotation + J → opacity/scale) ─
        # Each particle's TOTAL rotation is  R_rigid · R_loc:
        #   - R_rigid was applied in the rigid branch (μ' = μRᵀ + t)
        #     but SH was deferred to here (sh_new = sh in transform.py).
        #   - R_loc is the per-particle local deformation rotation from
        #     the physics backend (identity for rigid_contact, R_opt for
        #     PBD shape-matching, polar-decomp R for MPM).
        # SH is rotated EXACTLY ONCE here by the composed total rotation,
        # avoiding both double-rotation (if rigid did it too) and the
        # missing-R_rigid bug (using only R_loc).
        R_loc = phys_result.R_loc.reshape(B, K, N, 3, 3)
        J     = phys_result.J.reshape(B, K, N)
        if R_rigid is not None:
            R_total = R_rigid.unsqueeze(2).expand(B, K, N, 3, 3) @ R_loc   # [B,K,N,3,3]
        else:
            R_total = R_loc
        sh_new, opacity_new, scale_new = compensate_appearance(
            sh.reshape(B * K, N, -1),
            opacity.reshape(B * K, N, 1),
            scale.reshape(B * K, N, 3),
            R_total.reshape(B * K, N, 3, 3),
            J.reshape(B * K, N),
        )
        sh_new      = sh_new.reshape(B, K, N, -1)
        opacity_new = opacity_new.reshape(B, K, N, 1)
        scale_new   = scale_new.reshape(B, K, N, 3)

        return dict(
            mu=mu_new,
            cov=cov_new,
            sh=sh_new,
            opacity=opacity_new,
            scale=scale_new,
            route_probs=route_probs.reshape(B, K, -1),
            vfield_delta=vf_delta.reshape(B, K, N, 3),
        )
