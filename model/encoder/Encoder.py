"""
Top-level Encoder: perception -> motion -> discrete tokens.

Stages:
    1) ObjectDecomposer      : frames + gs_params (List[GSParameter]) -> slots, attn, canonical frames
    2) MotionEncoder         : slots + phi -> per-object per-window motion embeddings z
    3) ActionTokenizer       : z -> VQ codes {c_prim} with interpretable heads (l, h, xi, rho)

Note: Initial GSParameter is provided externally (pre-built 3DGS scene).

Input:
    frames : [B, V, T, C, H, W]

    B   - batch size
    V   - number of camera views
    T   - frames per clip
    C   - channels (3)
    H,W - resolution

"""
from __future__ import annotations
import math
from typing import List

import torch
import torch.nn as nn

from .ObjectDecomposer import ObjectDecomposer
from .MotionEncoder import MotionEncoder
from .ActionTokenizer import ActionTokenizer
from ..utils import GSParameter
# ===========================================================================
# Visibility mask from attention entropy
# ===========================================================================
def attention_to_visibility(
    attn: torch.Tensor,
    threshold: float = 0.2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Derive per-object, per-frame visibility confidence from attention masks.

    Sharp attention (low entropy)    → object visible  → confidence ≈ 1
    Diffuse attention (high entropy) → object occluded → confidence ≈ 0

    Args:
        attn:      [B, Tp, K, N_tokens]  — slot attention weights (softmax over K)
        threshold: confidence below this → considered occluded
        eps:       numerical stability

    Returns:
        mask: [B, Tp, K]  — float, 1.0 = visible, 0.0 = occluded
    """
    N_tokens = attn.shape[-1]
    max_entropy = math.log(max(N_tokens, 1))

    # ── Normalize over tokens so each slot has a proper distribution ──
    # Raw attn is softmax(dim=0) → sums to 1 over K, NOT over N.
    # We need a per-slot spatial distribution to compute meaningful entropy.
    p = attn / (attn.sum(dim=-1, keepdim=True) + eps)      # [B, Tp, K, N]

    # Entropy per object per frame (now bounded by log(N))
    entropy = -(p * (p + eps).log()).sum(dim=-1)            # [B, Tp, K]

    # Normalize to [0, 1]: 0 = sharp (visible), 1 = uniform (occluded)
    confidence = 1.0 - (entropy / max_entropy).clamp(0, 1) # [B, Tp, K]

    # Hard mask
    mask = (confidence > threshold).float()                 # [B, Tp, K]

    return mask


# ===========================================================================
# Top-level Encoder
# ===========================================================================
class Encoder(nn.Module):
    def __init__(
        self,
        gs_dimension: int,
        obj_cfg: dict,
        motion_cfg: dict,
        action_cfg: dict,
    ):
        super().__init__()
        
        # gs_state_dim = mu[3] + scale[3] + opacity[1] + sh[C_sh] + cov[4] = 11 + C_sh
        self.obj_enc = ObjectDecomposer(gs_dimension=gs_dimension, obj_param=obj_cfg)
        self.motion_enc = MotionEncoder(motion_cfg)
        self.action_enc = ActionTokenizer(action_cfg)
        self.visibility_threshold = float(obj_cfg.get('visibility_threshold', 0.2))

    def forward(
        self,
        frames: torch.Tensor,           # [B, V, T, C, H, W]
        tau: float,
        gs_params: List[GSParameter],    # len B, each flat [Ng, ...]
    ):

        # ---- Stage 1: Object Decomposition ----
        obj_out = self.obj_enc(frames, gs_states_list=gs_params, tau=tau)
        # obj_out: {slots, attn, assignment, logits, phi}

        # Compute visibility mask (Dataset-B: occlusions possible); When all slots are visible (Dataset-A), pass None for fast avg_pool path.
        mask = attention_to_visibility(obj_out["attn"],self.self.visibility_threshold)
        if mask.bool().all():
            mask = None

        # ---- Stage 2: Motion Encoding ----
        motion_out = self.motion_enc(
            slots=obj_out["slots"],
            phi=obj_out["phi"],
            mask=mask,
        )
        # motion_out: {z_motion, window_mask}

        # ---- Stage 3: Action Tokenization ----
        act_out = self.action_enc(z_motion=motion_out["z_motion"])
        # act_out: {tokens, quantized, vq_loss, recon, sub_quantized, physical_params}

        return {
            # Stage 1 → Executor (Stage 5), training.py
            "phi": obj_out["phi"],                          # CanonicalFrame: R_w2c [B, K, 3, 3], t_w2c [B, K, 3]
            "attn": obj_out["attn"],                        # attention weights [B, Tp, K, V*Hf*Wf]
            "assignment": obj_out["assignment"],            # List[Tensor[N_b, K]], len B (t=0 GS ↔ slot hard binding)
            "logits": obj_out["logits"],                    # List[Tensor[N_b, K]], len B

            # Stage 2 → training.py (recon target for Stage 3 decoder)
            "z_motion": motion_out["z_motion"],             # [B, T_act, K, motion_dim]

            # Stage 3 → Planner (Stage 4)
            "seq_tokens": act_out["tokens"],                # [B, T_act, K] codebook indices
            "seq_mask": motion_out["window_mask"],          # [B, T_act, K] bool | None

            # Stage 3 → Executor (Stage 5)
            "quantized": act_out["quantized"],              # fused features:[B, T, K, motion_dim]

            # Stage 3 → training.py (losses)
            "vq_loss": act_out["vq_loss"],
            "recon": act_out["recon"],
            "sub_quantized": act_out["sub_quantized"],       # "l": q_l, [B, T, K, d_l]; "h": q_h, [B, T, K, d_h]; "xi": q_xi, [B, T, K, d_xi]; "rho": q_rho, [B, T, K, d_rho]
            "physical_params": act_out["physical_params"],   # "translation", [B, T, K, 3]; "rotation", [B, T, K, 9]; "micro_rotation", [B, T, K, 3]; "deformation", [B, T, K, n] or None
        }
