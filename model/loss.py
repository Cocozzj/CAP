"""
loss.py — CAP Loss Suite (PDF f07d2c0a + fdfa011c).

Loss families
─────────────
A. Algebraic Structure (5):  L_clos, L_inv, L_eq, L_eq_cross, L_comm
B. Reconstruction (1):       L_rec      (Stage-0 / Stage-2)
C. Semantic Alignment (1):   L_NCE      (text ↔ task embedding)
D. Quantisation & Plan (4):  L_VQ_act, L_VQ_task, L_CVAE, L_hier
E. Regularisation (3):       L_Lip, L_entropy, L_phys

Stage activation
────────────────
  Stage-0  RIGID:   B(rec)+A(clos+inv+comm@0.01)+C(nce)+D(VQ)
  Stage-1  PHYSICS: above + E(phys+Lip)              [enc+planner frozen]
  Stage-2  FULL:    everything (eq@0.3, comm 0.01→0.1 ramp, KL β anneal)

Public API
──────────
    loss_fn = CAPLoss(cfg=...)
    losses  = loss_fn(model=model,
                      training_out=training_out,        # from CAPModel.forward
                      gt=gt_dict,                        # {"frames": ..., "depth": ..., "text": [...]}
                      stage=TrainingStage.RIGID)
    losses["total"].backward()

The ``training_out`` dict comes from ``CAPModel.training_forward`` and contains:
    encoder       — full Encoder forward output (incl. seq_tokens, physical_params,
                    sub_quantized, vq_loss, recon, phi, assignment, ...)
    planner       — full Planner forward output (logits, targets, mu/logvar,
                    vq_loss, h_task, recon_h_task, task_emb, v_proj)
    executor      — Executor.apply_sequence output (final SceneState + trajectory + aux_list)
    scene_state   — initial SceneState (before Executor)
    token_indices — [B, L] flat AR target tokens
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import SceneState, masked_mean


# ══════════════════════════════════════════════════════════════════════
# Default hyper-parameters (overridden by cfg/loss.yaml)
# ══════════════════════════════════════════════════════════════════════

DEFAULT_LOSS_CFG: Dict[str, Any] = {
    # A. Algebraic structure
    "lambda_clos":          0.5,
    "lambda_inv":           0.5,
    "lambda_eq":            0.3,
    "lambda_eq_cross":      0.3,
    "lambda_comm":          0.01,        # base; ramped 0.01 → 0.1 in Stage-2
    "lambda_comm_max":      0.1,

    # B. Reconstruction
    "lambda_rec":           1.0,
    "lambda_rec_mse":       0.2,
    "lambda_rec_lpips":     0.1,
    "lambda_depth":         0.5,

    # C. Semantic alignment
    "lambda_nce":           0.3,
    "nce_temperature":      0.07,

    # D. Quantisation + planner
    "lambda_vq_act":        1.0,         # Encoder VQ commitment
    "lambda_vq_task":       1.0,         # Planner TaskTokenizer VQ commitment
    "lambda_cvae_kl":       0.01,        # base; β anneal 0.01 → 1.0 in Stage-2
    "lambda_cvae_kl_max":   1.0,
    "lambda_cvae_recon":    1.0,
    "lambda_planner_ce":    1.0,         # AR cross-entropy
    "lambda_hier":          0.2,         # hierarchical consistency, Stage-2

    # E. Regularisation
    "lambda_lip":           0.1,
    "lip_target_C":         2.0,
    "lip_epsilon":          0.01,
    "lambda_entropy":       0.05,
    "entropy_H_min":        3.0,
    "entropy_H_max":        8.0,
    "lambda_physics":       0.5,
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

    Forward (per RigidTransform): R = exp(ξ) · R_h ; μ' = μRᵀ + ℓ.
    Inverse:  R⁻¹ = R_hᵀ · exp(−ξ) ; ℓ⁻¹ = −ℓ · R   (row-vec convention).

    Since RigidTransform always computes R = exp(ξ') · R_h', we encode the
    inverse as:
        R_h'  = R_hᵀ                           (transpose)
        ξ'    = −R_h ξ                          (adjoint conjugate)
        ⇒   exp(ξ') · R_h' = R_hᵀ · exp(−ξ) = R⁻¹
        ℓ'    = −ℓ · R                          (computed from ORIGINAL R)
        ρ'    = −ρ                              (linear deformation reversal)
    """
    R_h     = _ensure_R3x3(p["rotation"])                    # [B, K, 3, 3]
    xi      = p["micro_rotation"]                            # [B, K, 3]
    l       = p["translation"]                               # [B, K, 3]
    rho     = p.get("deformation", None)

    R_h_inv = R_h.transpose(-2, -1)                          # [B, K, 3, 3]
    # Adjoint conjugate of ξ
    xi_inv  = -torch.einsum("...ij,...j->...i", R_h, xi)     # [B, K, 3]

    # Forward total rotation R for translation inverse
    # NOTE: avoid importing exp_so3 here to keep loss.py independent;
    # we approximate ξ small (≤5°) → exp(ξ) ≈ I + skew(ξ).  For the inverse
    # algebraic loss this small-angle approximation suffices.
    R_total = R_h.clone()                                    # [B, K, 3, 3]  approximation
    l_inv   = -torch.einsum("...i,...ij->...j", l, R_total)

    inv = {
        "translation":    l_inv,
        "rotation":       R_h_inv.reshape(*R_h_inv.shape[:-2], 9),
        "micro_rotation": xi_inv,
        "deformation":    (-rho if rho is not None else None),
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
    """L_clos = d_M( E(g_a) ∘ E(g_b),  E(g_a ⊙̂ g_b) ).

    Closure under sequential composition.  We approximate the algebraic
    composition in the physical-parameter space (additive in (ℓ, ξ) and
    multiplicative in R_h), then compare to the sequential trajectory.
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    if T < 2:
        return scene.mu.new_zeros(())

    t = int(torch.randint(0, T - 1, (1,)).item())
    g_a = _slice_params(physical_params_seq, t)
    g_b = _slice_params(physical_params_seq, t + 1)

    # Sequential path: E(g_a) ∘ E(g_b)
    s1   = _apply_token_safe(executor, scene, g_a, enable_physics, task_context)
    s_seq = _apply_token_safe(executor, s1,   g_b, enable_physics, task_context)

    # Composed path: g_comp = (ℓ_a + R_a·ℓ_b,  R_a·R_b,  ξ_a + ξ_b,  ρ_a + ρ_b)
    R_a = _ensure_R3x3(g_a["rotation"])
    R_b = _ensure_R3x3(g_b["rotation"])
    R_comp = R_a @ R_b
    l_comp = g_a["translation"] + torch.einsum("...ij,...j->...i", R_a, g_b["translation"])
    xi_comp = g_a["micro_rotation"] + g_b["micro_rotation"]
    rho_comp = None
    if g_a.get("deformation") is not None and g_b.get("deformation") is not None:
        rho_comp = g_a["deformation"] + g_b["deformation"]
    g_comp = {
        "translation":    l_comp,
        "rotation":       R_comp.reshape(*R_comp.shape[:-2], 9),
        "micro_rotation": xi_comp,
        "deformation":    rho_comp,
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
    """Canonical-frame equivariance: actions executed in canonical space
    should agree with actions executed in world space (modulo Φ_o conjugation).

    Since Executor already does world → canonical → exec → world internally,
    this loss tests consistency by perturbing scene.phi and comparing outputs.
    For training-time efficiency, we approximate by checking idempotence under
    small Φ rotations (full Prop-3 conjugation handled by L_eq_cross below).
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
    """Two non-overlapping micro-actions on different objects should commute.

    L_comm = d_M( E(g_a) ∘ E(g_b),  E(g_b) ∘ E(g_a) )

    For simplicity we compare full-token sequential applications (relying on
    the algebraic structure of the codebook).  Theorem 3 bounds the gap.
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
    pred_frames: Optional[torch.Tensor],         # [B, V, T, 3, H, W] or None
    gt_frames:   Optional[torch.Tensor],
    pred_depth:  Optional[torch.Tensor] = None,
    gt_depth:    Optional[torch.Tensor] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, torch.Tensor]:
    """L_rec = MSE + LPIPS + depth (PDF §5.1).

    LPIPS is optional (requires lpips package).  Depth term only fires
    when both pred_depth and gt_depth are provided (Dataset-B with MiDaS).
    """
    cfg = cfg or DEFAULT_LOSS_CFG
    out = {}

    if pred_frames is None or gt_frames is None:
        # No rendered frames yet (training stage / loss disabled)
        zero = (pred_frames if pred_frames is not None else gt_frames
                if gt_frames is not None else torch.zeros(1)).new_zeros(())
        out["mse"]   = zero
        out["lpips"] = zero
        out["depth"] = zero
        out["rec_total"] = zero
        return out

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

    # Depth (single-view supervision)
    if pred_depth is not None and gt_depth is not None:
        out["depth"] = F.l1_loss(pred_depth, gt_depth)
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
    """L_hier: parse generated atomic seq back to a task token, compare to
    the original.  Active in Stage-2.

    Implementation: feed task_emb through the AR decoder, get generated
    sequence, then pass that sequence through TaskTokenizer to recover
    a task embedding, and compare cosine distance to the input task_emb.

    Ablation #3 (planner.use_task_token=False): the hierarchical round-trip
    is conceptually undefined (no task layer to round-trip THROUGH).  Returns
    0 in that case so the term contributes nothing to the total loss — the
    caller's lambda should also be set to 0 to keep the loss dict honest.
    """
    # No-hierarchical ablation → return 0 (no task layer to round-trip through)
    if getattr(planner, "task_tok", None) is None:
        return task_emb.new_zeros(())

    # Generate atomic seq from task_emb (deterministic for stability)
    cfg = dict(sampling_cfg or {})
    cfg["deterministic"] = True
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
) -> torch.Tensor:
    """Bound executor Jacobian: ‖E(g+δ) − E(g)‖ / ‖δ‖ ≤ C  (PDF §1.4).

    Implemented as a hinge: penalise (||ΔS|| / ||δ|| − C)_+².
    """
    T = next(v.shape[1] for v in physical_params_seq.values() if v is not None)
    t = int(torch.randint(0, T, (1,)).item())
    g = _slice_params(physical_params_seq, t)

    # Perturb micro_rotation
    g_pert = dict(g)
    perturb = eps * torch.randn_like(g["micro_rotation"])
    g_pert["micro_rotation"] = g["micro_rotation"] + perturb

    s_orig = _apply_token_safe(executor, scene, g,      enable_physics)
    s_pert = _apply_token_safe(executor, scene, g_pert, enable_physics)

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
    if not executor_aux:
        zero = torch.zeros(())
        return {"physics_vol": zero, "physics_contact": zero,
                "physics_energy": zero, "physics_total": zero}

    # Volume: penalise large vfield divergence (approximation)
    vol_terms = []
    for aux in executor_aux:
        vfd = aux.get("vfield_delta")
        if vfd is not None:
            vol_terms.append(vfd.flatten(1).norm(dim=-1).pow(2).mean())
    vol = torch.stack(vol_terms).mean() if vol_terms else torch.zeros(())

    # Contact: penalise t_used penetrating below ground (z < d_min)
    d_min = cfg["physics_contact_d_min"]
    contact_terms = []
    for aux in executor_aux:
        t_used = aux.get("t_used_world")
        if t_used is not None:
            contact_terms.append(F.relu(d_min - t_used[..., 2]).pow(2).mean())
    contact = torch.stack(contact_terms).mean() if contact_terms else torch.zeros(())

    # Energy: kinetic energy proxy from frame-to-frame translation magnitude
    energy_terms = []
    for aux in executor_aux:
        t_used = aux.get("t_used_world")
        if t_used is not None:
            energy_terms.append(t_used.flatten(1).norm(dim=-1).pow(2).mean())
    energy = torch.stack(energy_terms).mean() if energy_terms else torch.zeros(())

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
                          gt=gt_dict, stage=TrainingStage.RIGID,
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
        stage: int = 0,
        step: int = 0,
        total_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        out: Dict[str, torch.Tensor] = {}

        # Stage flag (TrainingStage constants: 0=RIGID, 1=PHYSICS, 2=FULL)
        is_rigid   = (stage == 0)
        is_physics = (stage == 1)
        is_full    = (stage == 2)

        enc_out  = training_out["encoder"]
        plan_out = training_out["planner"]
        exec_out = training_out["executor"]
        scene    = training_out["scene_state"]
        ppseq    = enc_out["physical_params"]
        seq_tok  = enc_out["seq_tokens"]
        K_codes  = model.encoder.action_enc.vq.num_codes

        # Task context for executor's deform branch
        task_emb = plan_out.get("task_emb")
        v_proj   = plan_out.get("v_proj")
        task_ctx = (model._expand_task_context(task_emb, scene.K)
                    if task_emb is not None else None)
        physics_on = (is_physics or is_full)

        # ── A. Algebraic structure ──────────────────────────────────
        out["L_clos"] = closure_loss(model.executor, scene, ppseq,
                                     enable_physics=physics_on, task_context=task_ctx)
        out["L_inv"]  = inverse_loss(model.executor, scene, ppseq,
                                     enable_physics=physics_on, task_context=task_ctx)

        if is_full:
            out["L_eq"]       = equivariance_loss(model.executor, scene, ppseq,
                                                  enable_physics=physics_on, task_context=task_ctx)
            out["L_eq_cross"] = equivariance_cross_object_loss(model.executor, scene, ppseq,
                                                               enable_physics=physics_on, task_context=task_ctx)
        else:
            zero = scene.mu.new_zeros(())
            out["L_eq"], out["L_eq_cross"] = zero, zero

        out["L_comm"] = commutator_loss(model.executor, scene, ppseq,
                                        enable_physics=physics_on, task_context=task_ctx)

        # ── B. Reconstruction (if GT frames provided) ───────────────
        if gt is not None and gt.get("frames") is not None:
            rec = reconstruction_loss(
                pred_frames=exec_out.get("rendered_frames"),
                gt_frames=gt.get("frames"),
                pred_depth=exec_out.get("rendered_depth"),
                gt_depth=gt.get("depth"),
                cfg=cfg,
            )
            out.update(rec)
        else:
            zero = scene.mu.new_zeros(())
            out["mse"], out["lpips"], out["depth"], out["rec_total"] = zero, zero, zero, zero

        # ── C. InfoNCE ──────────────────────────────────────────────
        out["L_NCE"] = infonce_loss(task_emb, v_proj, cfg["nce_temperature"]) \
                       if task_emb is not None else scene.mu.new_zeros(())

        # ── D. Quantisation + planner ───────────────────────────────
        out["L_VQ_act"] = enc_out.get("vq_loss", scene.mu.new_zeros(()))
        beta_kl = self._anneal(cfg["lambda_cvae_kl"], cfg["lambda_cvae_kl_max"],
                               step, total_steps) if is_full else cfg["lambda_cvae_kl"]
        cvae_d  = cvae_loss(plan_out, pad_id=model._pad_id,
                            kl_weight=beta_kl,
                            recon_weight=cfg["lambda_cvae_recon"],
                            ce_weight=cfg["lambda_planner_ce"])
        out.update(cvae_d)
        out["L_VQ_task"] = plan_out.get("vq_loss", scene.mu.new_zeros(()))

        if is_full:
            out["L_hier"] = hierarchical_loss(
                model.planner, model.encoder,
                text_embed=v_proj, task_emb=task_emb,
                sampling_cfg=model.planner.sampling_cfg,
            )
        else:
            out["L_hier"] = scene.mu.new_zeros(())

        # ── E. Regularisation ──────────────────────────────────────
        if physics_on:
            out["L_Lip"] = lipschitz_loss(model.executor, scene, ppseq,
                                          eps=cfg["lip_epsilon"], target_C=cfg["lip_target_C"],
                                          enable_physics=True)
        else:
            out["L_Lip"] = scene.mu.new_zeros(())

        if is_full:
            out["L_entropy"] = entropy_loss(seq_tok, K_codes,
                                            cfg["entropy_H_min"], cfg["entropy_H_max"])
        else:
            out["L_entropy"] = scene.mu.new_zeros(())

        if physics_on:
            phys = physics_loss(exec_out.get("aux_list", []), cfg=cfg)
            out.update(phys)
        else:
            zero = scene.mu.new_zeros(())
            out["physics_vol"], out["physics_contact"], out["physics_energy"], out["physics_total"] = \
                zero, zero, zero, zero

        # ── Stage-aware ramp for L_comm ─────────────────────────────
        comm_w = self._anneal(cfg["lambda_comm"], cfg["lambda_comm_max"], step, total_steps) \
                 if is_full else cfg["lambda_comm"]

        # ── Total ───────────────────────────────────────────────────
        total = (
            cfg["lambda_clos"]     * out["L_clos"]
          + cfg["lambda_inv"]      * out["L_inv"]
          + cfg["lambda_eq"]       * out["L_eq"]
          + cfg["lambda_eq_cross"] * out["L_eq_cross"]
          + comm_w                  * out["L_comm"]
          + cfg["lambda_rec"]      * out["rec_total"]
          + cfg["lambda_nce"]      * out["L_NCE"]
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
    """Cosine sim between input task_emb and re-encoded task_emb (eval-only)."""
    with torch.no_grad():
        loss = hierarchical_loss(planner, encoder, None, task_emb, sampling_cfg)
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
