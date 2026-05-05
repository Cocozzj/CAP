"""
Object Decomposition Pipeline (Stage 1).
    frames [B, V, T, C, H, W] + gs_params List[GSParameter] (len B, each flat [Ng, ...])
    -> CNNBackbone -> feats
    -> MultiViewTemporalSlotAttention -> slots [B, Tp, K, D], attn [B, Tp, K, V*Hf*Wf]
    -> trivial assignment (Gaussian k*N..(k+1)*N belongs to slot k by construction)
    -> phi computed via PCA on per-object Gaussian centers
"""
from __future__ import annotations
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import CNNBackbone, MultiViewTemporalSlotAttention
from ..utils import GSParameter, CanonicalFrame, compute_canonical_frame


# ===========================================================================
# Utilities
# ===========================================================================
def _freeze_eval(module: nn.Module) -> None:
    """Freeze a module for feature extraction."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    for m in module.modules():
        if hasattr(m, "gradient_checkpointing_disable"):
            try:
                m.gradient_checkpointing_disable()
            except Exception:
                pass
        if hasattr(m, "config") and hasattr(m.config, "gradient_checkpointing"):
            try:
                m.config.gradient_checkpointing = False
            except Exception:
                pass
# ===========================================================================
# SlotGSBinder: learned cross-attention Gaussian → object assignment
# ===========================================================================
class SlotGSBinder(nn.Module):
    """
    Assign N Gaussians to K object slots via cross-attention + Gumbel-softmax.
    - Training: Gumbel-softmax (hard forward, soft backward)
    - Inference: deterministic argmax
    """
    def __init__(
        self,
        slot_dim: int,
        gs_input_dim: int,
        gs_hidden_dim: int = 128,
    ):
        super().__init__()
        self.slot_dim = slot_dim

        # ── Gaussian encoder: [N, gs_input_dim] → [N, slot_dim] ──
        self.gs_encoder = nn.Sequential(
            nn.Linear(gs_input_dim, gs_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_hidden_dim, slot_dim),
        )
        self.gs_norm = nn.LayerNorm(slot_dim)

        # ── Slot projection: [K, slot_dim] → [K, slot_dim] ──
        self.slot_proj = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(inplace=True),
            nn.Linear(slot_dim, slot_dim),
        )
        self.slot_norm = nn.LayerNorm(slot_dim)

        self.scale = slot_dim ** -0.5

    def forward(
        self,
        slot_features: torch.Tensor,
        gs_state: GSParameter,
        tau: float = 1.0,
    ):
        """
        Args:
            slot_features: [K, slot_dim] — refined slot representations (t=0)
            gs_state:      class GSParameter:
                                mu: torch.Tensor          # [Ng, 3]     Gaussian centres (world)
                                cov: torch.Tensor         # [Ng, 4]     covariance (quaternion)
                                sh: torch.Tensor          # [Ng, C_sh]  SH coefficients
                                opacity: torch.Tensor     # [Ng, 1]     opacity  ∈ [0, 1]
                                scale: torch.Tensor       # [Ng, 3]     log-scale vector
            tau:           Gumbel-softmax temperature
        Returns:
            assignment [N, K], logits [N, K]
        """
        # ── Encode Gaussians ──
        gs_params = torch.cat([
            gs_state.mu, 
            gs_state.cov.reshape(gs_state.cov.shape[0], -1), 
            gs_state.scale,
            gs_state.opacity, 
            gs_state.sh,
        ], dim=-1)                                              # [N, gs_input_dim]

        gs_features = self.gs_encoder(gs_params)                # [N, slot_dim]
        gs_normed = self.gs_norm(gs_features)                   # [N, slot_dim]

        # ── Project slots ──
        slots_proj = self.slot_proj(slot_features)              # [K, slot_dim]
        slots_normed = self.slot_norm(slots_proj)               # [K, slot_dim]

        # ── Cross-attention: slots query, Gaussians are keys ──
        logits_kn = torch.matmul(
            slots_normed, gs_normed.transpose(0, 1)
        ) * self.scale                                          # [K, N]

        # ── Transpose: each Gaussian picks one slot ──
        logits = logits_kn.transpose(0, 1).contiguous()         # [N, K]

        # ── Gumbel-softmax assignment ──
        if self.training:
            assignment = F.gumbel_softmax(
                logits, tau=tau, hard=True, dim=-1
            )                                                   # [N, K]
        else:
            idx = logits.argmax(dim=-1)                         # [N]
            assignment = F.one_hot(
                idx, num_classes=logits.shape[-1]
            ).float()                                           # [N, K]

        return assignment, logits

    @staticmethod
    def get_object_gaussians(
        gs_state: GSParameter,
        assignment: torch.Tensor,
    ):
        """
        Extract per-object Gaussian subsets from hard assignment.

        Args:
            gs_state:   dict with Gaussian params, each [N, ...]
            assignment: [N, K] one-hot assignment

        Returns:
            List of K dicts, each containing:
                'mu', 'cov', 'scale', 'opacity', 'sh' — subset tensors
                'indices' — [n_k] indices into original N Gaussians
        """
        owners = assignment.argmax(dim=-1)                      # [N]
        K = assignment.shape[-1]

        subsets = []
        for k in range(K):
            mask = (owners == k)
            indices = mask.nonzero(as_tuple=False).squeeze(-1)
            if indices.numel() == 0:
                subsets.append({
                    'mu':      gs_state.mu.new_empty((0, gs_state.mu.shape[-1])),
                    'cov':    gs_state.cov.new_empty((0, gs_state.cov.shape[-1])),
                    'scale':   gs_state.scale.new_empty((0, gs_state.scale.shape[-1])),
                    'opacity': gs_state.opacity.new_empty((0, gs_state.opacity.shape[-1])),
                    'sh':      gs_state.sh.new_empty((0, gs_state.sh.shape[-1])),
                    'indices': indices,
                })
            else:
                subsets.append({
                    'mu':      gs_state.mu[mask],
                    'cov':    gs_state.cov[mask],
                    'scale':   gs_state.scale[mask],
                    'opacity': gs_state.opacity[mask],
                    'sh':      gs_state.sh[mask],
                    'indices': indices,
                })
        return subsets


# ===========================================================================
# Canonicalization: PCA-based per-object canonical frame
# ===========================================================================
class Canonicalization(nn.Module):
    """
    Compute canonical frames for K objects via PCA on their Gaussian centers.

    Returns:
    CanonicalFrame(R_w2c=R, t_w2c=t)
        R_w2c: [K, 3, 3]  — rotation matrices (world → canonical)
        t_w2c: [K, 3]     — centers of mass in world coords

    Transform helpers (standalone functions):
        world_to_canonical(x_w, R, t) = (x_w - t) @ R^T
        canonical_to_world(x_c, R, t) = x_c @ R + t
    """

    def __init__(self, min_points: int = 3, eps: float = 1e-6):
        super().__init__()
        self.min_points = min_points
        self.eps = eps

    def forward(
        self,
        object_subsets: List[Dict[str, torch.Tensor]],
    ):
        """
        Compute canonical frames for all K objects.

        Args:
            object_subsets: List of K dicts from SlotGSBinder.get_object_gaussians().
                            Each must have 'mu' [n_k, 3].

        Returns:
            CanonicalFrame with:
                R_w2c: [K, 3, 3]  — rotation matrices (world → canonical)
                t_w2c: [K, 3]     — centers of mass in world coords
        """
        all_R = []
        all_t = []
        for subset in object_subsets:
            mu = subset['mu']
            # Use opacity as PCA weights: emphasize opaque Gaussians,
            # suppress semi-transparent ones near boundaries.
            opacity = subset.get('opacity', None)
            weights = opacity.squeeze(-1) if opacity is not None else None  # [n_k]

            frame = compute_canonical_frame(
                positions=mu,
                weights=weights,
                min_points=self.min_points,
                eps=self.eps,
            )

            all_R.append(frame.R_w2c)
            all_t.append(frame.t_w2c)

        R = torch.stack(all_R, dim=0)                         # [K, 3, 3]
        t = torch.stack(all_t, dim=0)                         # [K, 3]
        return CanonicalFrame(R_w2c=R, t_w2c=t)

# ===========================================================================
# ObjectDecomposer: full Stage 1 pipeline
# ===========================================================================
class ObjectDecomposer(nn.Module):
    """
    Stage 1 encoder pipeline:
        frames [B, V, T, C, H, W] + gs_params [List[GSParameter], len B, each flat [Ng, ...]]
        → CNNBackbone → feats
        → MultiViewTemporalSlotAttention → slots [B, Tp, K, D], attn [B, Tp, K, V*Hf*Wf]
    """
    def __init__(self, gs_dimension: int, obj_param: dict):
        super().__init__()

        backbone_type = obj_param.get("backbone_type", "resnet34")

        # Stage 1.1: CNN Backbone
        self.backbone = CNNBackbone(
            backbone_type=backbone_type,
            resnet_param=obj_param.get("resnet34_param", {}),
            videomae_param=obj_param.get("videomae_param", {}),
            vit_param=obj_param.get("vit_param", {}),
        )
        self.freeze_backbone = bool(obj_param.get("freeze_backbone", True))
        if self.freeze_backbone:
            _freeze_eval(self.backbone)

        # Stage 1.2: Multi-View Temporal Slot Attention
        slotatt_param = dict(obj_param.get("slotatt_param", {}))
        self.slotattn = MultiViewTemporalSlotAttention(
            in_dim=self.backbone.out_channels,
            gs_state_dim = gs_dimension,
            **slotatt_param,
        )
        
        # Stage 1.3: SlotGSBinder
        self.binder = SlotGSBinder(
            slot_dim=slotatt_param.get("slot_dim", 128),
            gs_input_dim=gs_dimension,
            gs_hidden_dim=int(slotatt_param.get("gs_hidden_dim", 128)),
        )

        # Stage 1.4: Canonicalization
        self.canon_frame = Canonicalization()


    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(
        self,
        frames: torch.Tensor,             # [B, V, T, C, H, W]
        gs_states_list: List[GSParameter], # len B, each flat [Ng, ...]
        tau: float = 1.0,
    ) -> Dict[str, Any]:
    

        B = len(gs_states_list)

        # ---- Stage 1.1: CNN Backbone ----
        if self.freeze_backbone:
            with torch.no_grad():
                feats, _ = self.backbone(frames)
        else:
            feats, _ = self.backbone(frames)
        # feats: [B, V, T, D', Hf, Wf]

        # ---- Stage 1.2: Multi-View Temporal Slot Attention ----
        slots, attn = self.slotattn(feats, gs_states=gs_states_list, tau=tau)
        # slots: [B, Tp, K, slot_dim]
        # attn:  [B, Tp, K, V*Hf*Wf]

        # ---- Stage 1.3 + 1.4: Bind (t=0) and canonicalize per batch ----
        all_assignment = []
        all_logits = []
        all_R = []
        all_t = []

        for b in range(B):
            # 1.3 Bind using t=0 slots
            assignment, logit = self.binder(
                slots[b, 0], gs_states_list[b], tau=tau
            )
            all_assignment.append(assignment)                   # [N_b, K]
            all_logits.append(logit)                            # [N_b, K]

            # Extract per-object Gaussian subsets (used by canon, not stored)
            subsets = SlotGSBinder.get_object_gaussians(
                gs_states_list[b], assignment
            )

            # 1.4 Canonical frames via PCA
            phi_b = self.canon_frame(subsets)                   # CanonicalFrame
            all_R.append(phi_b.R_w2c)                           # [K, 3, 3]
            all_t.append(phi_b.t_w2c)                           # [K, 3]

        return {
            "slots":      slots,                                # [B, Tp, K, slot_dim]
            "attn":       attn,                                 # [B, Tp, K, V*Hf*Wf]
            "assignment": all_assignment,                       # List[Tensor[N_b, K]], len B (binding from t=0 slots)
            "logits":     all_logits,                           # List[Tensor[N_b, K]], len B
            "phi": CanonicalFrame(
                R_w2c=torch.stack(all_R, dim=0),                # [B, K, 3, 3]
                t_w2c=torch.stack(all_t, dim=0),                # [B, K, 3]
            ),
        }

