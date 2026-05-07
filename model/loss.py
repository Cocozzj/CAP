"""
loss.py — CAP Loss Suite (PDF f07d2c0a + fdfa011c).

Loss families
─────────────
A. Algebraic Structure (5):  L_clos, L_inv, L_eq, L_eq_cross, L_comm
B. Reconstruction (1):       L_rec
C. Semantic Alignment (1):   L_NCE      (text ↔ task embedding)
D. Quantisation & Plan (4):  L_VQ_act, L_VQ_task, L_CVAE, L_hier
E. Regularisation (3):       L_Lip, L_entropy, L_phys

Always-on terms (compute regardless of spec):
    L_clos, L_inv, L_comm, L_rec (if GT given), L_NCE, L_VQ_act, L_VQ_task, planner_total (CVAE recon + CE).

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SceneState, masked_mean


# ══════════════════════════════════════════════════════════════════════
# LossSpec — caller-controlled curriculum
# ══════════════════════════════════════════════════════════════════════

@dataclass
class LossSpec:
    """

    Physics-conditional (need physics rollouts):
        enable_physics:      pass enable_physics=True to clos/inv/comm
        enable_lipschitz:    add L_Lip
        enable_physics_loss: add L_phys (volume / contact / energy)

    Full-stage terms:
        enable_equiv:        add L_eq + L_eq_cross
        enable_hier:         add L_hier (text→task→atomic distillation)
        enable_entropy:      add L_entropy (codebook usage)

    Annealing schedules (over training step):
        anneal_cvae_kl:      ramp KL β from cfg.lambda_cvae_kl → ..._max
        anneal_comm:         ramp L_comm weight from cfg.lambda_comm → ..._max
    """
    enable_physics:      bool = False
    enable_lipschitz:    bool = False
    enable_physics_loss: bool = False
    enable_equiv:        bool = False
    enable_hier:         bool = False
    enable_entropy:      bool = False
    anneal_cvae_kl:      bool = False
    anneal_comm:         bool = False


# ══════════════════════════════════════════════════════════════════════
# Default hyper-parameters (overridden by cfg/loss.yaml)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_LOSS_CFG: Dict[str, Any] = {
    # ── Values aligned with PDF §模型训练配置 (page 13) ───────────────
    # A. Algebraic structure
    "lambda_clos":          0.5,         # PDF
    "lambda_inv":           0.5,         # PDF
    "lambda_eq":            0.5,         # PDF
    "lambda_eq_cross":      0.5,         # match lambda_eq
    "lambda_comm":          0.01,        # base; ramped 0.01 → 0.1 in Stage-2
    "lambda_comm_max":      0.1,

    # B. Reconstruction
    "lambda_rec":           0.2,         # PDF
    "lambda_rec_mse":       0.2,
    "lambda_rec_lpips":     0.1,         # PDF (lambda_lpips)
    "lambda_depth":         0.5,

    # C. Semantic alignment
    "lambda_nce":           0.1,         # PDF
    "nce_temperature":      0.07,

    # D. Quantisation + planner
    "lambda_vq_act":        1.0,         # standard VQ commitment
    "lambda_vq_task":       1.0,         # standard VQ commitment
    "lambda_cvae_kl":       0.01,        # base; β anneal 0.01 → 0.1 (PDF endpoint)
    "lambda_cvae_kl_max":   0.1,         # PDF: β=0.1 fixed; we land here at end of Stage-2
    "lambda_cvae_recon":    1.0,
    "lambda_planner_ce":    1.0,         # AR cross-entropy
    "lambda_hier":          0.2,

    # E. Regularisation
    "lambda_lip":           0.01,    # see configs/loss.yaml comment for rationale
    "lip_target_C":         2.0,
    "lip_epsilon":          0.01,
    "lambda_entropy":       0.05,
    "entropy_H_min":        3.0,
    "entropy_H_max":        8.0,
    "lambda_physics":       0.1,         # PDF
    "lambda_physics_vol":   0.5,
    "lambda_physics_contact": 1.0,
    "lambda_physics_energy":  1.0,
    "physics_contact_d_min":  0.01,
}


# ══════════════════════════════════════════════════════════════════════
# §0  Scene-distance metric (mask-aware)
# ══════════════════════════════════════════════════════════════════════

def scene_distance(state_a: SceneState, state_b: SceneState) -> torch.Tensor:
    """Mean L2 displacement over Gaussian centres, mask-aware.

    d_M(state_a, state_b) = mean_{real Gaussians} ||μ_a − μ_b||₂

    Returns: scalar (averaged over [B, K, N_real])
    """
    diff = (state_a.mu - state_b.mu).norm(dim=-1)            # [B, K, N]
    # Defensive: even with masked_mean now NaN-safe (model/utils.py),
    # an entire batch/slot row of NaN would still produce NaN.  Replace
    # any residual NaN with a large but finite penalty so training fights
    # them down via gradient instead of locking up.
    diff = torch.nan_to_num(diff, nan=10.0, posinf=10.0, neginf=10.0)
    if state_a.mask is not None:
        return masked_mean(diff, state_a.mask, dim=-1).mean()
    return diff.mean()


# ══════════════════════════════════════════════════════════════════════
# §1  Helpers — slice / invert / compose physical_params
# ══════════════════════════════════════════════════════════════════════

def _slice_params(params: Dict[str, torch.Tensor], t: int) -> Dict[str, torch.Tensor]:
    """Slice [B, T_act, K, ...] → [B, K, ...] at time t."""
    return {k: (v[:, t] if v is not None else None) for k, v in params.items()}


def _ensure_R3x3(rot: torch.Tensor) -> torch.Tensor:
    """[..., 9] or [..., 3, 3] → [..., 3, 3]."""
    if rot.shape[-1] == 9 and rot.shape[-2] != 3:
        return rot.reshape(*rot.shape[:-1], 3, 3)
    return rot


def _invert_params(p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Construct SE(3)-inverse of an action token's physical_params.

    Forward (per RigidTransform.transform.py): R = exp(ξ) · R_h, μ' = μ Rᵀ + ℓ
                                              (row-vector convention)

    Inverse must satisfy:  exp(ξ_inv) · R_h_inv = R⁻¹ = R_hᵀ · exp(−ξ)

    Decomposition:
        R_h_inv = R_hᵀ                          (transpose)
        exp(ξ_inv) = R_hᵀ · exp(−ξ) · R_h
                   = exp(skew(R_hᵀ · (−ξ)))     (SO(3) conjugation identity)
        ⇒ ξ_inv = −R_hᵀ · ξ                     ← NOT −R_h · ξ (was a bug)

    Translation in row-vec form:  μ = (μ' − ℓ) R = μ' R − ℓ R
        ⇒ ℓ_inv = −ℓ · R                        (computed from ORIGINAL R)

    For the deformation slot ρ (9-d named physics tuple — Physics-Plugin PDF
    §1.3): there is NO group-theoretic "inverse" of (E, ν, ρ_m, F, μ, damping,
    dt) — these are physical parameters, not group elements.  We approximate
    by negating the FORCE component only (the only additive-composable slot)
    and zeroing the others, so applying g then g_inv cancels the *force* but
    leaves material/contact unchanged — a partial round-trip that's better
    than naively negating all 9 dims (which previously inverted Young's
    modulus etc., which is physically meaningless).
    """
    R_h     = _ensure_R3x3(p["rotation"])                    # [B, K, 3, 3]
    xi      = p["micro_rotation"]                            # [B, K, 3]
    l       = p["translation"]                               # [B, K, 3]
    rho     = p.get("deformation", None)

    R_h_inv = R_h.transpose(-2, -1)                          # [B, K, 3, 3]

    # Adjoint conjugate of ξ:  ξ_inv = −R_hᵀ · ξ
    # einsum "...ji,...j->...i" computes (R_hᵀ ξ)_i = R_h_ji · ξ_j ✓
    xi_inv  = -torch.einsum("...ji,...j->...i", R_h, xi)     # [B, K, 3]

    # Translation inverse:  ℓ_inv = −ℓ · R  ≈  −ℓ · R_h  (small ξ ≤ 5° approx)
    R_total = R_h.clone()
    l_inv   = -torch.einsum("...i,...ij->...j", l, R_total)

    # Deformation: only invert the additive force slot (indices 3:6).
    # E / ν / ρ_m / μ / damping / dt do NOT have group inverses.
    rho_inv = None
    if rho is not None:
        rho_inv = rho.clone()
        rho_inv[..., 3:6] = -rho[..., 3:6]                   # force ↦ -force
        # other slots stay equal so material/contact "cancel" trivially

    inv = {
        "translation":    l_inv,
        "rotation":       R_h_inv.reshape(*R_h_inv.shape[:-2], 9),
        "micro_rotation": xi_inv,
        "deformation":    rho_inv,
    }
    return inv


def _apply_token_safe(
    executor,
    scene: SceneState,
    params: Dict[str, torch.Tensor],
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> SceneState:
    """Wrapper: apply a single token, return only the new SceneState."""
    new_scene, _aux = executor.apply_token(
        scene=scene,
        physical_params=params,
        enable_physics=enable_physics,
        task_context=task_context,
    )
    return new_scene


# ══════════════════════════════════════════════════════════════════════
# A.1  Closure loss — L_clos (Theorem 1)
# ══════════════════════════════════════════════════════════════════════

def closure_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],   # [B, T_act, K, ...]
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_clos = d_M( E(g_b) ∘ E(g_a),  E(g_a ⊙̂ g_b) ).

    Closure under sequential composition.  Apply g_a then g_b to the scene,
    and compare to applying the algebraically-composed action g_comp once.

    Composition rule (row-vector convention used by RigidTransform: μ' = μ Rᵀ + ℓ):
        After g_a:    μ → μ R_aᵀ + ℓ_a
        After g_b:    μ → μ R_aᵀ R_bᵀ + ℓ_a R_bᵀ + ℓ_b
                      = μ (R_b R_a)ᵀ + ℓ_a R_bᵀ + ℓ_b

    So:
        R_comp = R_b · R_a       (NOT R_a · R_b — that was a column-vec bug)
        ℓ_comp = ℓ_a · R_bᵀ + ℓ_b

    For micro-rotation ξ ≤ 5° we use the small-angle approximation
    ξ_comp ≈ ξ_a + ξ_b (BCH error  O(‖ξ_a × ξ_b‖) ~ 0.4° per step — acceptable).

    For ρ (9-d named physics tuple, Physics-Plugin PDF §1.3): physical
    parameters do NOT have a group-composition rule (you can't "compose"
    Young's moduli or densities).  We pass ρ_a (just the first action's
    physics) so g_comp still triggers the deform branch — the closure check
    reduces to the SE(3) part, which is what algebraic structure is about.
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    if T < 2:
        return scene.mu.new_zeros(())

    t = int(torch.randint(0, T - 1, (1,)).item())
    g_a = _slice_params(physical_params_seq, t)
    g_b = _slice_params(physical_params_seq, t + 1)

    # Sequential path: apply g_a first, then g_b
    s1    = _apply_token_safe(executor, scene, g_a, enable_physics, task_context)
    s_seq = _apply_token_safe(executor, s1,    g_b, enable_physics, task_context)

    # Composed path — algebraically combine in row-vec convention
    R_a = _ensure_R3x3(g_a["rotation"])
    R_b = _ensure_R3x3(g_b["rotation"])
    R_comp = torch.einsum("...ij,...jk->...ik", R_b, R_a)            # R_b @ R_a
    # ℓ_comp = ℓ_a · R_bᵀ + ℓ_b  ⇔  einsum("...i,...ji->...j", ℓ_a, R_b)
    l_comp = (torch.einsum("...i,...ji->...j", g_a["translation"], R_b)
              + g_b["translation"])
    xi_comp = g_a["micro_rotation"] + g_b["micro_rotation"]          # small-angle BCH approx
    g_comp = {
        "translation":    l_comp,
        "rotation":       R_comp.reshape(*R_comp.shape[:-2], 9),
        "micro_rotation": xi_comp,
        # ρ does NOT compose group-theoretically — pass ρ_a so the deform
        # branch fires once with comparable physics, but closure is really
        # only checking the SE(3) algebraic structure here.
        "deformation":    g_a.get("deformation"),
    }
    s_comp = _apply_token_safe(executor, scene, g_comp, enable_physics, task_context)

    return scene_distance(s_seq, s_comp)


# ══════════════════════════════════════════════════════════════════════
# A.2  Inverse loss — L_inv (Theorem 2)
# ══════════════════════════════════════════════════════════════════════

def inverse_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L_inv = ½ [ d_M(E(g) ∘ E(ĝ⁻¹), id)  +  d_M(E(ĝ⁻¹) ∘ E(g), id) ]."""
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    t = int(torch.randint(0, T, (1,)).item())
    g     = _slice_params(physical_params_seq, t)
    g_inv = _invert_params(g)

    # forward then inverse
    s1   = _apply_token_safe(executor, scene, g,     enable_physics, task_context)
    s1_r = _apply_token_safe(executor, s1,    g_inv, enable_physics, task_context)
    # inverse then forward
    s2   = _apply_token_safe(executor, scene, g_inv, enable_physics, task_context)
    s2_r = _apply_token_safe(executor, s2,    g,     enable_physics, task_context)

    return 0.5 * (scene_distance(s1_r, scene) + scene_distance(s2_r, scene))


# ══════════════════════════════════════════════════════════════════════
# A.3  Equivariance loss — L_eq (action commutes with canonical frame)
# ══════════════════════════════════════════════════════════════════════

def equivariance_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Φ-PERTURBATION STABILITY (NOT the strict PDF L_eq formula).

    PDF Physics-Plugin §1.4 defines equivariance as conjugation:
        Φ_o⁻¹ ∘ E_o(g) ∘ Φ_o ≈ E_c(g)
    i.e. executing g in object frame vs canonical frame should agree modulo
    the Φ_o coordinate change.

    Implementing that strictly requires Executor to expose a "force-execute-
    in-canonical" entry point separate from its current world→can→exec→world
    pipeline.  Until that refactor lands, this loss approximates by:
      1. Run E(g) on the original scene → s_orig
      2. Perturb scene.phi.R_w2c by ~3° random rotation
      3. Run E(g) on the perturbed scene → s_pert
      4. L = d_M(s_orig, s_pert)

    What this trains: E should be CONTINUOUSLY DEPENDENT on Φ — small Φ
    perturbations produce small output changes.  This is a *necessary*
    condition for true conjugation equivariance but not *sufficient*.

    Strict cross-object equivariance (PDF §4.1 Prop 3) is handled by
    ``equivariance_cross_object_loss`` which uses ``executor.transfer_object``.
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    t = int(torch.randint(0, T, (1,)).item())
    g = _slice_params(physical_params_seq, t)

    # Original execution
    s_orig = _apply_token_safe(executor, scene, g, enable_physics, task_context)

    # Perturb scene.phi by a small rotation, execute same token
    B, K = scene.phi.R_w2c.shape[:2]
    angle = 0.05  # ~3 degrees
    axis  = F.normalize(torch.randn(B, K, 3, device=scene.mu.device), dim=-1)
    # Rodrigues for small angle: R ≈ I + θ·skew(axis)
    skew = torch.zeros(B, K, 3, 3, device=scene.mu.device, dtype=scene.mu.dtype)
    skew[..., 0, 1] = -axis[..., 2] * angle
    skew[..., 0, 2] =  axis[..., 1] * angle
    skew[..., 1, 0] =  axis[..., 2] * angle
    skew[..., 1, 2] = -axis[..., 0] * angle
    skew[..., 2, 0] = -axis[..., 1] * angle
    skew[..., 2, 1] =  axis[..., 0] * angle
    R_pert = torch.eye(3, device=scene.mu.device, dtype=scene.mu.dtype) + skew
    new_phi_R = R_pert @ scene.phi.R_w2c

    from .utils import CanonicalFrame
    pert_scene = SceneState(
        mu=scene.mu, cov=scene.cov, sh=scene.sh,
        opacity=scene.opacity, scale=scene.scale,
        phi=CanonicalFrame(R_w2c=new_phi_R, t_w2c=scene.phi.t_w2c),
        mask=scene.mask, R_obj_world=scene.R_obj_world,
    )
    s_pert = _apply_token_safe(executor, pert_scene, g, enable_physics, task_context)

    # Distance should be small (action is equivariant to small phi perturbations)
    return scene_distance(s_orig, s_pert)


# ══════════════════════════════════════════════════════════════════════
# A.4  Cross-object equivariance — L_eq_cross (PDF §4.1 Prop 3)
# ══════════════════════════════════════════════════════════════════════

def equivariance_cross_object_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply action on src object's frame vs. transferred to tgt object's frame.

    Uses ``executor.transfer_object`` internally.  Picks two random objects
    per batch and checks that the transferred trajectory matches direct
    execution scaled by the canonical-frame conjugation.
    """
    K = scene.K
    if K < 2:
        return scene.mu.new_zeros(())

    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    t = int(torch.randint(0, T, (1,)).item())
    g = _slice_params(physical_params_seq, t)

    src_k = int(torch.randint(0, K, (1,)).item())
    tgt_k = int(torch.randint(0, K, (1,)).item())
    if src_k == tgt_k:
        tgt_k = (src_k + 1) % K

    # Transferred execution: src action applied on tgt geometry/frame
    s_xfer, _ = executor.transfer_object(
        scene=scene, physical_params=g,
        src_k=src_k, tgt_k=tgt_k,
        enable_physics=enable_physics, task_context=task_context,
    )

    # Direct execution then read out tgt slot
    s_direct = _apply_token_safe(executor, scene, g, enable_physics, task_context)

    # Compare displacement trajectories at the tgt slot
    diff = (s_xfer.mu[:, tgt_k] - s_direct.mu[:, tgt_k]).norm(dim=-1)  # [B, N]
    if scene.mask is not None:
        return masked_mean(diff, scene.mask[:, tgt_k], dim=-1).mean()
    return diff.mean()


# ══════════════════════════════════════════════════════════════════════
# A.5  Commutator loss — L_comm
# ══════════════════════════════════════════════════════════════════════

def commutator_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],
    enable_physics: bool = False,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """ANY-PAIR COMMUTATIVITY PENALTY (weaker than PDF "对易子探针").

        L_comm = d_M( E(g_a) ∘ E(g_b),  E(g_b) ∘ E(g_a) )

    PDF Physics-Plugin §1.4 specifies the commutator probe should target
    operations that "本应交换/抵消" (should commute or cancel) — typically
    actions on disjoint object slots, or an action paired with its partial
    inverse.  The strict probe should:
      - pick K_a ≠ K_b slot indices and apply g_a only on K_a, g_b only on K_b
      - then E(g_a)∘E(g_b) and E(g_b)∘E(g_a) are mathematically required to
        agree because the actions act on disjoint Gaussians

    This implementation instead takes ANY two consecutive timestep actions
    and applies them to ALL slots — so it penalises NON-commutativity even
    when non-commutativity is the correct behaviour (e.g. rotate-then-translate
    vs translate-then-rotate on the same object).

    Effect: acts as a soft prior pushing the codebook toward more
    commutative-like actions.  Useful as a low-weight regulariser
    (lambda_comm anneals 0.01→0.1) but NOT a strict commutator probe.
    Upgrade to true cross-slot probe when ``executor.apply_token`` gains a
    per-slot mask argument.
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    if T < 2:
        return scene.mu.new_zeros(())
    t = int(torch.randint(0, T - 1, (1,)).item())
    g_a = _slice_params(physical_params_seq, t)
    g_b = _slice_params(physical_params_seq, t + 1)

    s_ab = _apply_token_safe(executor,
                _apply_token_safe(executor, scene, g_a, enable_physics, task_context),
                g_b, enable_physics, task_context)
    s_ba = _apply_token_safe(executor,
                _apply_token_safe(executor, scene, g_b, enable_physics, task_context),
                g_a, enable_physics, task_context)
    return scene_distance(s_ab, s_ba)


# ══════════════════════════════════════════════════════════════════════
# B.  Reconstruction loss
# ══════════════════════════════════════════════════════════════════════

def reconstruction_loss(
    pred_frames:        Optional[torch.Tensor],   # [B, V, T_pred, 3, H, W] or None
    gt_frames:          Optional[torch.Tensor],   # [B, V, T_gt,   3, H, W]
    pred_depth:         Optional[torch.Tensor] = None,
    gt_depth:           Optional[torch.Tensor] = None,
    cfg:                Optional[Dict[str, Any]] = None,
    rendered_timesteps: Optional[List[int]] = None,   # which T_gt indices pred covers
    T_total:            Optional[int] = None,         # len(trajectory) — for index mapping
) -> Dict[str, torch.Tensor]:
    """L_rec = MSE + LPIPS + depth (PDF §5.1).

    Renderer typically renders a SPARSE subset of timesteps (e.g. [initial,
    final]) to keep training cost manageable.  ``rendered_timesteps`` lists
    those indices in [0, T_total]; we subsample ``gt_frames`` to match before
    computing pixel losses.

    LPIPS is optional (requires lpips package).  Depth term only fires
    when both pred_depth and gt_depth are provided.
    """
    cfg = cfg or DEFAULT_LOSS_CFG
    out = {}

    if pred_frames is None or gt_frames is None:
        # No rendered frames yet (gsplat missing / cameras not passed / loss disabled)
        anchor = pred_frames if pred_frames is not None else gt_frames
        zero = anchor.new_zeros(()) if anchor is not None else torch.zeros(())
        out["mse"]       = zero
        out["lpips"]     = zero
        out["depth"]     = zero
        out["rec_total"] = zero
        return out

    # ── Align pred (T_pred) and gt (T_gt) along the time axis ──
    # pred_frames: [B, V, T_pred, 3, H, W]   gt_frames: [B, V, T_gt, 3, H, W]
    T_pred = pred_frames.shape[2]
    T_gt   = gt_frames.shape[2]
    if T_pred != T_gt:
        if rendered_timesteps is not None and T_total is not None and T_total > 0:
            # Map render-side index (0..T_total) → gt-frame index (0..T_gt-1)
            scale = (T_gt - 1) / max(T_total, 1)
            gt_idx = [min(int(round(t * scale)), T_gt - 1) for t in rendered_timesteps]
        else:
            # Fallback: take first/last/uniform stride
            if T_pred == 1:
                gt_idx = [T_gt - 1]
            elif T_pred == 2:
                gt_idx = [0, T_gt - 1]
            else:
                stride = max(T_gt // T_pred, 1)
                gt_idx = list(range(0, T_gt, stride))[:T_pred]
        gt_frames = gt_frames[:, :, gt_idx]
        # Only the per-frame (6-D) depth tensor has a time axis at dim 2.
        # The 4-D static format used by DatasetB (MiDaS first-frame depth) has
        # shape [B, V, H, W] — slicing dim 2 here would corrupt the H axis
        # (e.g. produce [B, V, T_pred, W] instead of [B, V, H, W]).
        if pred_depth is not None and gt_depth is not None and gt_depth.dim() == 6:
            gt_depth = gt_depth[:, :, gt_idx]

    out["mse"] = F.mse_loss(pred_frames, gt_frames)

    # LPIPS — optional dep; skipped silently if not installed
    try:
        import lpips
        if not hasattr(reconstruction_loss, "_lpips_net"):
            reconstruction_loss._lpips_net = lpips.LPIPS(net="alex").to(pred_frames.device)
        # LPIPS expects [B, 3, H, W] in [-1, 1]
        flat_pred = (pred_frames.flatten(0, -4) * 2 - 1).clamp(-1, 1)
        flat_gt   = (gt_frames.flatten(0, -4) * 2 - 1).clamp(-1, 1)
        out["lpips"] = reconstruction_loss._lpips_net(flat_pred, flat_gt).mean()
    except Exception:
        out["lpips"] = pred_frames.new_zeros(())

    # Depth supervision.
    # Two GT shapes are supported:
    #   per-frame: [B, V, T_gt, 1, H, W]      (subsample to render timesteps, full match)
    #   static:    [B, V, H, W]               (DatasetB MiDaS — single map for the
    #                                          INITIAL frame only).  In this case
    #                                          we only score pred_depth's first
    #                                          rendered timestep against it.
    if pred_depth is not None and gt_depth is not None:
        if gt_depth.dim() == 6:
            # Already aligned at frame level — gt_depth was sub-sampled with frames
            out["depth"] = F.l1_loss(pred_depth, gt_depth)
        elif gt_depth.dim() == 4:
            # Static initial-frame depth (DatasetB).  Compare to pred_depth[:, :, 0]
            # only if the renderer actually rendered the initial timestep — if it
            # didn't (e.g. ``rendered_timesteps=[15, 29]``), the t=0 slice is bogus
            # and we silently skip the depth term.
            if rendered_timesteps is not None and rendered_timesteps[0] != 0:
                out["depth"] = pred_frames.new_zeros(())
            else:
                pred0 = pred_depth[:, :, 0, 0]                     # [B, V, H, W]
                # MiDaS / DepthAnything output is *scale-ambiguous* (relative
                # disparity).  pred0 is rendered metric-style depth.  Their raw
                # ranges differ by 10-100×, so plain L1 explodes and dominates
                # ``rec_total``.  Normalise each sample by its own median before
                # comparison — turns this into a scale-invariant L1.
                # Scale-and-shift invariant L1 (DPT-style, Ranftl 2021).
                #
                # Mini-run TB observation: pure median normalisation alone
                # left ``loss/depth`` at ~75 because the gsplat camera-space
                # depth and DepthAnything *relative* depth differ by both
                # scale AND offset (and pred0 can even be negative for points
                # behind the rendering frustum).  We need to also remove the
                # per-sample shift before scaling.
                #
                # Approach: subtract the median (robust to outliers), then
                # divide by the mean absolute deviation (a.k.a. MAD).  If
                # pred = a*gt + b for any (a, b), this yields exactly equal
                # post-normalisation tensors, so L1 → 0 — exactly the
                # invariance we want against arbitrary scene-scale offsets.
                #
                # Both medians and the *pred-side* MAD are detached; we want
                # the pred to learn structure relative to gt, not to game
                # its own normaliser.
                B = pred0.shape[0]
                eps = 1e-3
                p_med = pred0.detach().flatten(1).median(dim=1).values.view(B, 1, 1, 1)
                g_med = gt_depth.flatten(1).median(dim=1).values.view(B, 1, 1, 1)
                p_centered = pred0    - p_med
                g_centered = gt_depth - g_med
                p_scale = p_centered.detach().abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                g_scale = g_centered.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
                out["depth"] = F.l1_loss(p_centered / p_scale,
                                         g_centered / g_scale)
        else:
            out["depth"] = pred_frames.new_zeros(())
    else:
        out["depth"] = pred_frames.new_zeros(())

    out["rec_total"] = (
        cfg["lambda_rec_mse"]   * out["mse"]
        + cfg["lambda_rec_lpips"] * out["lpips"]
        + cfg["lambda_depth"]   * out["depth"]
    )
    return out


# ══════════════════════════════════════════════════════════════════════
# C.  InfoNCE — task ↔ text alignment
# ══════════════════════════════════════════════════════════════════════

def infonce_loss(
    task_emb: torch.Tensor,                       # [B, task_dim]
    v_proj:   Optional[torch.Tensor],             # [B, task_dim] or None
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE between task embeddings and text embeddings."""
    if v_proj is None:
        return task_emb.new_zeros(())
    t = F.normalize(task_emb, dim=-1)
    v = F.normalize(v_proj,   dim=-1)
    sim = t @ v.T / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return 0.5 * (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels))


# ══════════════════════════════════════════════════════════════════════
# D.  Quantisation + planner losses
# ══════════════════════════════════════════════════════════════════════

def cvae_loss(
    planner_out: Dict[str, Any],
    pad_id: int = -1,
    kl_weight: float = 0.5,
    recon_weight: float = 1.0,
    ce_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Planner CE + KL + task-recon (Planner.training_forward output)."""
    logits  = planner_out["logits"]                  # [B, L, K_prim]
    targets = planner_out["targets"]                 # [B, L]
    mu      = planner_out["mu"]                      # [B, z_dim]
    logvar  = planner_out["logvar"]                  # [B, z_dim]
    h_task  = planner_out["h_task"]                  # [B, task_dim]
    recon_h = planner_out["recon_h_task"]            # [B, task_dim]

    B, L, V = logits.shape
    ce = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1),
                         ignore_index=pad_id, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    recon = F.mse_loss(recon_h, h_task.detach())

    return {
        "planner_ce":     ce,
        "planner_kl":     kl,
        "planner_recon":  recon,
        "planner_total":  ce_weight * ce + kl_weight * kl + recon_weight * recon,
    }


def hierarchical_loss(
    planner,
    encoder,
    text_embed: Optional[torch.Tensor],            # [B, task_dim] (optional)
    task_emb:   torch.Tensor,                      # [B, task_dim]
    sampling_cfg: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """L_hier — ONE-SIDED round-trip consistency (anchors TaskTokenizer + codebook
    to whatever the AR decoder currently emits).  Active in Stage-2 / FULL.

    Pipeline::

        task_emb  ─┬→ build_cond_mem ─→ AR.decode_generate ─→ seq[B,L]  (no_grad)
                   │                                              │
                   │                                              ▼
                   │                       F.embedding(seq, codebook.weight)
                   │                                              │
                   │                                              ▼
                   └→ ──── compare ←──── TaskTokenizer.encode  h_task_re

    GRADIENT FLOWS to (i.e. these modules ARE trained by L_hier):
      - TaskTokenizer.encode parameters
      - Action VQ codebook weight (via F.embedding lookup on generated indices)
      - Planner's task_emb encoding path (through build_cond_mem)

    GRADIENT DOES NOT FLOW to:
      - AR decoder transformer weights — discrete sampling inside
        ``decode_generate`` is wrapped in @torch.no_grad and we don't apply
        Gumbel relaxation.  This is intentional: the AR decoder is already
        trained densely via cross-entropy (``lambda_planner_ce=1.0``) and
        CVAE reconstruction (``lambda_cvae_recon=1.0``) — both far stronger
        signals than the L_hier round-trip would provide.

    The semantics is therefore "force TaskTokenizer + codebook to be
    inverse-consistent with the AR decoder's current outputs", NOT "train
    the AR decoder to produce re-encodable sequences".  If you ever need
    the latter, replace ``decode_generate`` with a soft-token (Gumbel /
    softmax-then-codebook-mix) path so gradients can flow through AR.

    Ablation #3 (planner.use_task_token=False): the hierarchical round-trip
    is conceptually undefined (no task layer to round-trip THROUGH).  Returns
    0 in that case — caller's lambda should also be 0 for an honest loss dict.
    """
    # No-hierarchical ablation → return 0 (no task layer to round-trip through)
    if getattr(planner, "task_tok", None) is None:
        return task_emb.new_zeros(())

    # Generate atomic seq from task_emb — force GREEDY sampling for stability.
    # NOTE: ``cfg["deterministic"]`` is NOT read by Sampler (it reads
    # ``cfg["strategy"]`` which defaults to "multinomial").  We must set
    # strategy explicitly, otherwise multinomial sampling on possibly
    # non-finite logits triggers CUDA device-side asserts.
    cfg = dict(sampling_cfg or {})
    cfg["strategy"]      = "greedy"
    cfg["temperature"]   = 1.0
    cfg["num_samples"]   = 1
    z = task_emb.new_zeros(task_emb.size(0), planner.cvae.z_dim)
    cond_mem, mem_mask = planner.cvae.build_cond_mem(task_emb=task_emb, z_task=z)
    gen = planner.cvae.decode_generate(
        cond_mem=cond_mem, sampling_cfg=cfg, mem_mask=mem_mask,
    )
    seq = gen["sequences"]                          # [B, L_out]

    # Re-encode through TaskTokenizer
    safe_idx = seq.clamp(0, encoder.action_enc.vq.num_codes - 1)
    embeds = F.embedding(safe_idx, encoder.action_enc.vq.codebook.weight)  # [B, L, atomic_dim]
    _task_id, _task_emb_re, h_task_re, _vq = planner.task_tok.encode(
        token_indices=seq, token_embeds=embeds,
    )
    return 1.0 - F.cosine_similarity(h_task_re, task_emb, dim=-1).mean()


# ══════════════════════════════════════════════════════════════════════
# E.  Regularisation losses
# ══════════════════════════════════════════════════════════════════════

def lipschitz_loss(
    executor,
    scene: SceneState,
    physical_params_seq: Dict[str, torch.Tensor],
    eps: float = 0.01,
    target_C: float = 2.0,
    enable_physics: bool = True,
    task_context: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bound executor Jacobian: ‖E(g+δ) − E(g)‖ / ‖δ‖ ≤ C  (PDF §1.4).

    Implemented as a hinge: penalise (||ΔS|| / ||δ|| − C)_+².
    Threads ``task_context`` through to keep the perturbation comparison
    under the same conditioning as the rest of the loss suite.
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    t = int(torch.randint(0, T, (1,)).item())
    g = _slice_params(physical_params_seq, t)

    # Perturb micro_rotation
    g_pert = dict(g)
    perturb = eps * torch.randn_like(g["micro_rotation"])
    g_pert["micro_rotation"] = g["micro_rotation"] + perturb

    s_orig = _apply_token_safe(executor, scene, g,      enable_physics, task_context)
    s_pert = _apply_token_safe(executor, scene, g_pert, enable_physics, task_context)

    delta = (s_pert.mu - s_orig.mu).flatten(1).norm(dim=-1)        # [B]
    perturb_norm = perturb.flatten(1).norm(dim=-1).clamp(min=1e-8)  # [B]
    ratio = delta / perturb_norm
    return F.relu(ratio - target_C).pow(2).mean()


def entropy_loss(
    seq_tokens: torch.Tensor,                 # [B, T_act, K] long
    num_codes: int,
    H_min: float = 3.0,
    H_max: float = 8.0,
) -> torch.Tensor:
    """Codebook utilisation entropy — Stage-2 only.

    Penalise entropy outside [H_min, H_max] (bits).
    Promotes diverse but not uniform codebook usage.
    """
    flat = seq_tokens.reshape(-1)
    counts = torch.bincount(flat.clamp(min=0, max=num_codes - 1), minlength=num_codes).float()
    p = counts / (counts.sum() + 1e-8)
    H = -(p * (p + 1e-12).log2()).sum()                                # entropy in bits

    over  = F.relu(H - H_max)
    under = F.relu(H_min - H)
    return (over + under).pow(2)


def physics_loss(
    executor_aux: List[Dict[str, Any]],         # apply_sequence aux_list
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """L_phys = volume-preservation + contact (no penetration) + energy.

    Aggregates over all timesteps in the trajectory.  Uses physical quantities
    from the aux dicts (route_probs, vfield_delta, R_used, t_used).

    Returns dict with components and total.
    """
    cfg = cfg or DEFAULT_LOSS_CFG

    # Find device from any aux tensor so zero placeholders go to the right device
    def _zero():
        for aux in executor_aux:
            for v in aux.values():
                if isinstance(v, torch.Tensor):
                    return v.new_zeros(())
        return torch.zeros(())              # only when aux is fully empty (eval-only)

    if not executor_aux:
        z = _zero()
        return {"physics_vol": z, "physics_contact": z,
                "physics_energy": z, "physics_total": z}

    # Volume: penalise large vfield divergence (approximation)
    vol_terms = []
    for aux in executor_aux:
        vfd = aux.get("vfield_delta")
        if vfd is not None:
            vol_terms.append(vfd.flatten(1).norm(dim=-1).pow(2).mean())
    vol = torch.stack(vol_terms).mean() if vol_terms else _zero()

    # Contact: penalise t_used penetrating below ground (z < d_min)
    d_min = cfg["physics_contact_d_min"]
    contact_terms = []
    for aux in executor_aux:
        t_used = aux.get("t_used_world")
        if t_used is not None:
            contact_terms.append(F.relu(d_min - t_used[..., 2]).pow(2).mean())
    contact = torch.stack(contact_terms).mean() if contact_terms else _zero()

    # Energy: kinetic energy proxy from frame-to-frame translation magnitude
    energy_terms = []
    for aux in executor_aux:
        t_used = aux.get("t_used_world")
        if t_used is not None:
            energy_terms.append(t_used.flatten(1).norm(dim=-1).pow(2).mean())
    energy = torch.stack(energy_terms).mean() if energy_terms else _zero()

    total = (cfg["lambda_physics_vol"]     * vol
             + cfg["lambda_physics_contact"] * contact
             + cfg["lambda_physics_energy"] * energy)
    return {"physics_vol": vol, "physics_contact": contact,
            "physics_energy": energy, "physics_total": total}


# ══════════════════════════════════════════════════════════════════════
# CAPLoss — top-level orchestrator
# ══════════════════════════════════════════════════════════════════════

class CAPLoss(nn.Module):
    """Stage-aware loss aggregator.

    Usage::

        loss_fn = CAPLoss(cfg=cfg.get("loss", {}))
        losses  = loss_fn(model=model, training_out=training_out,
                          gt=gt_dict, spec=LossSpec(),
                          step=0, total_steps=10000)
        losses["total"].backward()
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.cfg = {**DEFAULT_LOSS_CFG, **(cfg or {})}

    # ── Anneal helpers ────────────────────────────────────────────
    def _anneal(self, base: float, max_v: float, step: int, total: int) -> float:
        if total <= 0:
            return base
        frac = min(max(step / float(total), 0.0), 1.0)
        return base + (max_v - base) * frac

    def forward(
        self,
        *,
        model,                                       # CAPModel
        training_out: Dict[str, Any],
        gt: Optional[Dict[str, Any]] = None,         # {"frames", "depth", "text"}
        spec: Optional[LossSpec] = None,             # which terms to compute / anneal
        step: int = 0,
        total_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        cfg  = self.cfg
        spec = spec or LossSpec()                    # default: everything OFF
        out: Dict[str, torch.Tensor] = {}

        enc_out  = training_out["encoder"]
        plan_out = training_out["planner"]
        exec_out = training_out["executor"]
        scene    = training_out["scene_state"]
        ppseq    = enc_out["physical_params"]
        seq_tok  = enc_out["seq_tokens"]
        K_codes  = model.encoder.action_enc.vq.num_codes
        zero     = scene.mu.new_zeros(())

        # Task context for executor's deform branch
        task_emb = plan_out.get("task_emb")
        v_proj   = plan_out.get("v_proj")
        task_ctx = (model._expand_task_context(task_emb, scene.K)
                    if task_emb is not None else None)
        phys_on  = spec.enable_physics

        # ── A. Algebraic structure ──────────────────────────────────
        out["L_clos"] = closure_loss(model.executor, scene, ppseq,
                                     enable_physics=phys_on, task_context=task_ctx)
        out["L_inv"]  = inverse_loss(model.executor, scene, ppseq,
                                     enable_physics=phys_on, task_context=task_ctx)
        out["L_comm"] = commutator_loss(model.executor, scene, ppseq,
                                        enable_physics=phys_on, task_context=task_ctx)

        if spec.enable_equiv:
            out["L_eq"]       = equivariance_loss(model.executor, scene, ppseq,
                                                  enable_physics=phys_on, task_context=task_ctx)
            out["L_eq_cross"] = equivariance_cross_object_loss(model.executor, scene, ppseq,
                                                               enable_physics=phys_on, task_context=task_ctx)
        else:
            out["L_eq"], out["L_eq_cross"] = zero, zero

        # ── B. Reconstruction (if GT frames provided) ───────────────
        if gt is not None and gt.get("frames") is not None:
            out.update(reconstruction_loss(
                pred_frames        = exec_out.get("rendered_frames"),
                gt_frames          = gt.get("frames"),
                pred_depth         = exec_out.get("rendered_depth"),
                gt_depth           = gt.get("depth"),
                cfg                = cfg,
                rendered_timesteps = exec_out.get("rendered_timesteps"),
                T_total            = exec_out.get("rendered_T_total"),
            ))
        else:
            out["mse"], out["lpips"], out["depth"], out["rec_total"] = zero, zero, zero, zero

        # ── C. InfoNCE ──────────────────────────────────────────────
        # NCE on BOTH post-VQ (task_emb) and pre-VQ (h_task) for robustness
        # to codebook collapse.  Post-VQ aligns the discrete task tokens
        # with text; pre-VQ aligns the continuous projection with text and
        # is the one that actually keeps lang.proj_head diverse when the
        # codebook itself collapses (as observed in our seed-0/1/2 runs:
        # all 128 entries cos-sim > 0.99).  See configs/loss.yaml.
        h_task_for_nce = plan_out.get("h_task")        # [B, task_dim] pre-VQ
        out["L_NCE"]       = (
            infonce_loss(task_emb, v_proj, cfg["nce_temperature"])
            if (task_emb is not None and v_proj is not None) else zero
        )
        out["L_NCE_preVQ"] = (
            infonce_loss(h_task_for_nce, v_proj, cfg["nce_temperature"])
            if (h_task_for_nce is not None and v_proj is not None) else zero
        )

        # ── D. Quantisation + planner ───────────────────────────────
        out["L_VQ_act"]  = enc_out.get("vq_loss", zero)
        out["L_VQ_task"] = plan_out.get("vq_loss") if plan_out.get("vq_loss") is not None else zero

        # Skip CVAE losses entirely if Planner forward was skipped
        # (caller passed run_planner=False → plan_out has no logits/targets/mu/...)
        if plan_out and "logits" in plan_out:
            beta_kl = (self._anneal(cfg["lambda_cvae_kl"], cfg["lambda_cvae_kl_max"], step, total_steps)
                       if spec.anneal_cvae_kl else cfg["lambda_cvae_kl"])
            out.update(cvae_loss(plan_out, pad_id=model._pad_id,
                                 kl_weight=beta_kl,
                                 recon_weight=cfg["lambda_cvae_recon"],
                                 ce_weight=cfg["lambda_planner_ce"]))
        else:
            out["planner_ce"]    = zero
            out["planner_kl"]    = zero
            out["planner_recon"] = zero
            out["planner_total"] = zero

        if spec.enable_hier:
            out["L_hier"] = hierarchical_loss(
                model.planner, model.encoder,
                text_embed=v_proj, task_emb=task_emb,
                sampling_cfg=model.planner.sampling_cfg,
            )
        else:
            out["L_hier"] = zero

        # ── E. Regularisation ──────────────────────────────────────
        if spec.enable_lipschitz:
            out["L_Lip"] = lipschitz_loss(model.executor, scene, ppseq,
                                          eps=cfg["lip_epsilon"], target_C=cfg["lip_target_C"],
                                          enable_physics=phys_on, task_context=task_ctx)
        else:
            out["L_Lip"] = zero

        if spec.enable_entropy:
            out["L_entropy"] = entropy_loss(seq_tok, K_codes,
                                            cfg["entropy_H_min"], cfg["entropy_H_max"])
        else:
            out["L_entropy"] = zero

        if spec.enable_physics_loss:
            out.update(physics_loss(exec_out.get("aux_list", []), cfg=cfg))
        else:
            out["physics_vol"], out["physics_contact"], out["physics_energy"], out["physics_total"] = \
                zero, zero, zero, zero

        # ── L_comm weight ramp ──────────────────────────────────────
        comm_w = (self._anneal(cfg["lambda_comm"], cfg["lambda_comm_max"], step, total_steps)
                  if spec.anneal_comm else cfg["lambda_comm"])

        # ── Per-component NaN guard before summing ──────────────────
        # Any single NaN component would poison ``total`` → backward
        # produces NaN gradients → optimizer.step makes ALL params NaN.
        # Sanitize each component (NaN → 0) so:
        #   - finite components still contribute their correct gradient
        #   - NaN components contribute zero gradient (skipped this step)
        #   - training self-heals instead of locking into NaN forever
        # The original NaN value is still surfaced in the per-component log
        # via the train_epoch diagnostic (it inspects ``losses`` dict before
        # this guard via per-component reads, but here we replace before sum).
        _SANITIZE_KEYS = (
            "L_clos", "L_inv", "L_eq", "L_eq_cross", "L_comm",
            "rec_total", "L_NCE", "L_VQ_act", "L_VQ_task",
            "planner_total", "L_hier", "L_Lip", "L_entropy", "physics_total",
        )
        for k in _SANITIZE_KEYS:
            v = out.get(k)
            if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                out[k] = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)

        # ── Total ───────────────────────────────────────────────────
        total = (
            cfg["lambda_clos"]     * out["L_clos"]
          + cfg["lambda_inv"]      * out["L_inv"]
          + cfg["lambda_eq"]       * out["L_eq"]
          + cfg["lambda_eq_cross"] * out["L_eq_cross"]
          + comm_w                  * out["L_comm"]
          + cfg["lambda_rec"]      * out["rec_total"]
          + cfg["lambda_nce"]      * out["L_NCE"]
          + cfg.get("lambda_nce_preVQ", 0.0) * out.get("L_NCE_preVQ", zero)
          + cfg["lambda_vq_act"]   * out["L_VQ_act"]
          + cfg["lambda_vq_task"]  * out["L_VQ_task"]
          + out["planner_total"]                         # already weighted internally
          + cfg["lambda_hier"]     * out["L_hier"]
          + cfg["lambda_lip"]      * out["L_Lip"]
          + cfg["lambda_entropy"]  * out["L_entropy"]
          + cfg["lambda_physics"]  * out["physics_total"]
        )
        out["total"] = total
        return out


# ══════════════════════════════════════════════════════════════════════
# Monitoring helpers
# ══════════════════════════════════════════════════════════════════════

def hierarchical_accuracy(
    planner, encoder, task_emb: torch.Tensor,
    sampling_cfg: Optional[Dict[str, Any]] = None,
) -> float:
    """Cosine sim between input task_emb and re-encoded task_emb (eval-only).

    Forces ``planner.eval()`` for the call to prevent the TaskTokenizer EMA
    update path from firing (would silently mutate the codebook even though
    this is an "accuracy" metric).  Restores the original mode afterwards.
    """
    was_training = planner.training
    planner.eval()
    try:
        with torch.no_grad():
            loss = hierarchical_loss(planner, encoder, None, task_emb, sampling_cfg)
    finally:
        if was_training:
            planner.train()
    return float(1.0 - loss.item())          # convert distance back to similarity


def codebook_utilisation(seq_tokens: torch.Tensor, num_codes: int) -> Dict[str, float]:
    """Stats over how the action codebook is used (eval-only)."""
    flat = seq_tokens.reshape(-1)
    counts = torch.bincount(flat.clamp(min=0, max=num_codes - 1), minlength=num_codes).float()
    p = counts / (counts.sum() + 1e-8)
    H = -(p * (p + 1e-12).log2()).sum().item()
    used = int((counts > 0).sum().item())
    return {"entropy_bits": H, "num_used_codes": used,
            "fraction_used": used / num_codes,
            "max_count": int(counts.max().item())}
