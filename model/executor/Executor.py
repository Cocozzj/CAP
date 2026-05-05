"""
executor.py — Top-level Executor for the CAP system.

Orchestrates the rigid + deformation branches in canonical space, then maps
back to world space.  Operates exclusively on the padded ``SceneState``
data structure (no internal group/ungroup).

Token interface:
    Receives ``physical_params`` directly from Encoder.ActionTokenizer.
    Keys (per-step shapes; sequence variants add a leading T dim):
        translation:    [B, K, 3]                     ℓ
        rotation:       [B, K, 3, 3]  or  [B, K, 9]   R_h (already SO(3))
        micro_rotation: [B, K, 3]                     ξ ∈ so(3)
        deformation:    [B, K, rho_dim]  or  None     ρ (None → physics disabled)

Public API
──────────
  apply_token        : single-step execution (one atomic action per object)
  apply_sequence     : sequential execution of T atomic actions
  transfer_object    : cross-object action transfer (PDF §4.1 Prop. 3)

Internal data flow per step
───────────────────────────
  scene (world)
      ↓ to_canonical(mu, cov, phi)
  scene_can (canonical)
      ↓ RigidTransform(physical_params, mask)
  scene_can_rigid
      ↓ DeformSim(physical_params["deformation"], mask, enable_physics)
  scene_can_new
      ↓ from_canonical(...)
  scene_world_new
      ↓ project_spd (defensive SPD projection on cov)
      ↓ update R_obj_world tracking
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .rigid.transform import RigidTransform
from .deform.sim import DeformSim

from ..utils import (
    SceneState,
    CanonicalFrame,
    to_canonical,
    from_canonical,
    project_spd,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers — shape normalisation for physical_params
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_rotation_3x3(rotation: torch.Tensor) -> torch.Tensor:
    """Accept rotation as either [..., 9] or [..., 3, 3]; return [..., 3, 3]."""
    if rotation.shape[-1] == 9 and rotation.dim() >= 2 and rotation.shape[-2] != 3:
        return rotation.reshape(*rotation.shape[:-1], 3, 3)
    if rotation.shape[-2:] == (3, 3):
        return rotation
    raise ValueError(
        f"Unexpected rotation shape {tuple(rotation.shape)}; "
        f"expected [..., 9] or [..., 3, 3]."
    )


def _slice_physical_params(physical_params: Dict[str, torch.Tensor],
                           t: int) -> Dict[str, torch.Tensor]:
    """Slice [B, T, K, ...] tensors at time index t → [B, K, ...]."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in physical_params.items():
        out[k] = v[:, t] if v is not None else None
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Executor
# ═══════════════════════════════════════════════════════════════════════════

class Executor(nn.Module):
    """Physics-aware Executor.

    Composes RigidTransform (always on) + DeformSim (optional physics
    plugin) on a padded SceneState.  Receives structured physical_params
    directly from Encoder.ActionTokenizer — no internal token re-parsing.
    """

    def __init__(
        self,
        rho_dim: int = 16,
        task_dim: int = 0,
        use_tfn_residual: bool = True,
        # ── TFN residual hidden dims (PDF f07d2c0a residual config) ──
        tfn_scalar_dim: int = 16,
        tfn_vector_dim: int = 4,
        # ── Deform branch ──
        param_mode: str = "logeuclid",
        router_temperature: float = 1.0,
        router_hard: bool = False,
        max_delta_mu: float = 1.0,
        # ── Optional sub-component config dicts (each None → defaults) ──
        # NOTE: PBD's per-step iteration counts (n_iters, n_substeps) flow into
        # RhoParser via rho_parser_cfg, NOT into ShapeMatchingPBD directly —
        # that's why this signature has no pbd_backend_cfg.
        rho_parser_cfg:    Optional[Dict] = None,
        rigid_contact_cfg: Optional[Dict] = None,
        # ── Ablation: restrict to a subset of physics backends ──
        enabled_backends:  Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.rho_dim = rho_dim

        # Rigid branch (always active)
        self.rigid = RigidTransform(
            use_tfn_residual=use_tfn_residual,
            tfn_scalar_dim=tfn_scalar_dim,
            tfn_vector_dim=tfn_vector_dim,
        )

        # Deformation branch (active when enable_physics=True)
        self.deform = DeformSim(
            rho_dim=rho_dim,
            task_dim=task_dim,
            param_mode=param_mode,
            router_temperature=router_temperature,
            router_hard=router_hard,
            max_delta_mu=max_delta_mu,
            rho_parser_cfg=rho_parser_cfg,
            rigid_contact_cfg=rigid_contact_cfg,
            enabled_backends=enabled_backends,
        )

    # ──────────────────────────────────────────────────────────────────
    # §1  apply_token  — single atomic action step
    # ──────────────────────────────────────────────────────────────────

    def apply_token(
        self,
        scene: SceneState,
        physical_params: Dict[str, torch.Tensor],     # per-step (no T dim)
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,  # [B, K, task_dim]
    ) -> Tuple[SceneState, dict]:
        """Apply ONE atomic action token to the scene.

        Args:
            scene:           SceneState to act on (padded + mask)
            physical_params: dict with keys translation [B,K,3],
                             rotation [B,K,3,3] or [B,K,9],
                             micro_rotation [B,K,3],
                             deformation [B,K,rho_dim] or None
            enable_physics:  activate DeformSim physics backends
            task_context:    [B, K, task_dim] or None

        Returns:
            (new_scene, aux_dict)
        """
        # ── Normalise & validate input ──────────────────────────────
        translation    = physical_params["translation"]                    # [B, K, 3]
        rotation       = _ensure_rotation_3x3(physical_params["rotation"]) # [B, K, 3, 3]
        micro_rotation = physical_params["micro_rotation"]                 # [B, K, 3]
        deformation    = physical_params.get("deformation", None)          # [B, K, rho_dim] or None

        # If deformation is missing, we cannot run physics — fall back gracefully
        if deformation is None:
            if enable_physics:
                warnings.warn(
                    "enable_physics=True but physical_params['deformation'] is None; "
                    "falling back to rigid-only execution.",
                    stacklevel=2,
                )
                enable_physics = False
            B, K = translation.shape[:2]
            deformation = torch.zeros(
                B, K, self.rho_dim,
                device=translation.device, dtype=translation.dtype,
            )

        # rho_summary for the rigid-branch TFN residual: 4 PDF-defined physical
        # scalars decoded from ρ — Young's modulus E, density ρ_m, Coulomb
        # friction μ, and velocity damping (PDF f07d2c0a §1.3 / fdfa011c 行 197).
        # We pre-decode here so the rigid branch can condition its quantisation-
        # compensation residual on the same physical quantities the deform branch
        # will later use, then thread the decoded params back into ``self.deform``
        # to avoid running RhoParser twice on the same ρ (matters for both perf
        # and correctness if RhoParser ever gains stochastic layers).
        B, K = deformation.shape[:2]
        rho_flat = deformation.reshape(B * K, -1)
        tc_flat  = (task_context.reshape(B * K, -1)
                    if task_context is not None else None)
        rho_summary_flat, preview_params = self.deform.physics_summary(
            rho_flat, tc_flat,
        )
        rho_summary = rho_summary_flat.reshape(B, K, -1)        # auto-fit summary dim
        
        # ── Step 1: world → canonical ───────────────────────────────
        mu_can, cov_can = to_canonical(scene.mu, scene.cov, scene.phi)

        # ── Step 2: Rigid SE(3) (in canonical space) ────────────────
        rigid_out = self.rigid(
            mu=mu_can, cov=cov_can,
            sh=scene.sh, scale=scene.scale,
            translation=translation,
            rotation=rotation,
            micro_rotation=micro_rotation,
            rho_summary=rho_summary,
            mask=scene.mask,
        )

        # ── Step 3: Deformation (physics plugin or fallback) ────────
        # Pass the pre-decoded physics params back so DeformSim doesn't
        # call RhoParser a second time (encapsulation + perf + determinism).
        deform_out = self.deform(
            mu=rigid_out["mu"],
            cov=rigid_out["cov"],
            sh=rigid_out["sh"],
            opacity=scene.opacity,
            scale=rigid_out["scale"],
            rho=deformation,
            enable_physics=enable_physics,
            task_context=task_context,
            R_rigid=rigid_out["R_used"],   # used by both paths for SH rotation
            mask=scene.mask,
            precomputed_params=preview_params,
        )

        mu_can_new  = deform_out["mu"]
        cov_can_new = deform_out["cov"]
        sh_new      = deform_out["sh"]
        opacity_new = deform_out["opacity"]
        scale_new   = deform_out["scale"]

        # ── Step 4: canonical → world ───────────────────────────────
        mu_world, cov_world = from_canonical(mu_can_new, cov_can_new, scene.phi)
        cov_world = project_spd(cov_world)

        # ── Step 5: update accumulated world-space rotation ─────────
        # World-space rotation = Φ_o⁻¹ ∘ R_tok ∘ Φ_o = R_c2w R_tok R_w2c
        R_tok = rigid_out["R_used"]                                      # [B, K, 3, 3]
        R_c2w = scene.phi.R_c2w                                          # [B, K, 3, 3]
        R_w2c = scene.phi.R_w2c
        R_obj_new = R_c2w @ R_tok @ R_w2c @ scene.R_obj_world

        # Re-orthonormalise (SVD requires float32 under AMP)
        _orig_dtype = R_obj_new.dtype
        if _orig_dtype in (torch.float16, torch.bfloat16):
            R_obj_new = R_obj_new.float()
        U, _, Vt = torch.linalg.svd(R_obj_new)
        R_obj_new = (U @ Vt).to(_orig_dtype)

        # ── Step 6: build new SceneState ────────────────────────────
        new_scene = SceneState(
            mu=mu_world,
            cov=cov_world,
            sh=sh_new,
            opacity=opacity_new,
            scale=scale_new,
            phi=scene.phi,                      # phi is persistent across steps
            mask=scene.mask,                    # mask is persistent
            R_obj_world=R_obj_new,
        )

        # ── Aux info (rotations, translations, route probs, etc.) ───
        # World-space translation: rotate canonical t_used into world coords
        t_used_can   = rigid_out["t_used"]                               # [B, K, 3]
        t_used_world = torch.einsum('...i,...ij->...j', t_used_can, R_c2w)

        aux = dict(
            R_used=R_tok,
            t_used_can=t_used_can,
            t_used_world=t_used_world,
            route_probs=deform_out.get("route_probs"),
            vfield_delta=deform_out.get("vfield_delta"),
        )
        return new_scene, aux

    # ──────────────────────────────────────────────────────────────────
    # §2  apply_sequence  — execute T atomic actions sequentially
    # ──────────────────────────────────────────────────────────────────

    def apply_sequence(
        self,
        scene: SceneState,
        physical_params_seq: Dict[str, torch.Tensor],   # tensors are [B, T, K, ...]
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,    # [B, K, task_dim] (constant across T)
    ) -> Tuple[SceneState, List[SceneState], List[dict]]:
        """Sequentially apply T atomic actions to the scene.

        Args:
            scene:                 initial SceneState
            physical_params_seq:   each tensor has leading [B, T, K, ...] shape
            enable_physics:        bool
            task_context:          [B, K, task_dim] (constant across timesteps)

        Returns:
            (final_scene, trajectory, aux_list)
            - final_scene : SceneState after T steps
            - trajectory  : list of T intermediate SceneStates
            - aux_list    : list of T dicts (R_used, t_used, route_probs, …)
        """
        # Determine T from any per-step tensor (use the first non-None)
        T = None
        for v in physical_params_seq.values():
            if v is not None and v.dim() >= 3:
                T = v.shape[1]
                break
        if T is None:
            raise ValueError(
                "apply_sequence: physical_params_seq has no time dimension"
            )

        trajectory: List[SceneState] = []
        aux_list:   List[dict]       = []

        current = scene
        for t in range(T):
            params_t = _slice_physical_params(physical_params_seq, t)
            current, aux = self.apply_token(
                scene=current,
                physical_params=params_t,
                enable_physics=enable_physics,
                task_context=task_context,
            )
            trajectory.append(current)
            aux_list.append(aux)

        return current, trajectory, aux_list

    # ──────────────────────────────────────────────────────────────────
    # §3  transfer_object  — cross-object action transfer (PDF §4.1 Prop. 3)
    # ──────────────────────────────────────────────────────────────────

    def transfer_object(
        self,
        scene: SceneState,
        physical_params: Dict[str, torch.Tensor],     # per-step (no T dim)
        src_k: int,
        tgt_k: int,
        enable_physics: bool = True,
        task_context: Optional[torch.Tensor] = None,
    ) -> Tuple[SceneState, dict]:
        """Transfer an action from object src_k to object tgt_k.

        Mathematically:
            E_{tgt}(g) ≈ Φ_tgt⁻¹ ∘ E_canon(g) ∘ Φ_tgt

        Procedure:
          1. Build a 1-object slice of the scene at tgt_k (with phi_tgt)
          2. Slice physical_params at src_k (the source action)
          3. Apply the source action on the target geometry in tgt's canonical frame
          4. Scatter the result back into the full scene
        """
        K = scene.K
        if not (0 <= src_k < K and 0 <= tgt_k < K):
            raise IndexError(
                f"src_k={src_k} or tgt_k={tgt_k} out of range for K={K} objects"
            )

        # ── Build 1-object SceneState for tgt_k ─────────────────────
        tgt_scene = SceneState(
            mu=scene.mu[:, tgt_k:tgt_k + 1],
            cov=scene.cov[:, tgt_k:tgt_k + 1],
            sh=scene.sh[:, tgt_k:tgt_k + 1],
            opacity=scene.opacity[:, tgt_k:tgt_k + 1],
            scale=scene.scale[:, tgt_k:tgt_k + 1],
            phi=CanonicalFrame(
                R_w2c=scene.phi.R_w2c[:, tgt_k:tgt_k + 1],
                t_w2c=scene.phi.t_w2c[:, tgt_k:tgt_k + 1],
            ),
            mask=scene.mask[:, tgt_k:tgt_k + 1],
            R_obj_world=scene.R_obj_world[:, tgt_k:tgt_k + 1],
        )

        # ── Slice physical_params at src_k (action params) ──────────
        params_src: Dict[str, torch.Tensor] = {}
        for key, val in physical_params.items():
            if val is None:
                params_src[key] = None
            elif val.dim() >= 2:
                params_src[key] = val[:, src_k:src_k + 1]
            else:
                params_src[key] = val

        tc_1 = task_context[:, tgt_k:tgt_k + 1] if task_context is not None else None

        # ── Apply source action on target's geometry/canonical frame ─
        new_tgt, aux = self.apply_token(
            scene=tgt_scene,
            physical_params=params_src,
            enable_physics=enable_physics,
            task_context=tc_1,
        )

        # ── Scatter back into the full scene ────────────────────────
        new_mu      = scene.mu.clone()
        new_cov     = scene.cov.clone()
        new_sh      = scene.sh.clone()
        new_opacity = scene.opacity.clone()
        new_scale   = scene.scale.clone()
        new_R_obj   = scene.R_obj_world.clone()

        new_mu[:, tgt_k:tgt_k + 1]      = new_tgt.mu
        new_cov[:, tgt_k:tgt_k + 1]     = new_tgt.cov
        new_sh[:, tgt_k:tgt_k + 1]      = new_tgt.sh
        new_opacity[:, tgt_k:tgt_k + 1] = new_tgt.opacity
        new_scale[:, tgt_k:tgt_k + 1]   = new_tgt.scale
        new_R_obj[:, tgt_k:tgt_k + 1]   = new_tgt.R_obj_world

        new_scene = SceneState(
            mu=new_mu, cov=new_cov, sh=new_sh,
            opacity=new_opacity, scale=new_scale,
            phi=scene.phi,
            mask=scene.mask,
            R_obj_world=new_R_obj,
        )
        return new_scene, aux
