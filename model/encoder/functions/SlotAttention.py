from ...utils import GSParameter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

# =============================================================================
# Slot Initialization: Learned Base + Gated GS Offset
# =============================================================================
class SlotInit(nn.Module):
    """
    Initialize K slot vectors using learned base slots + gated GS-conditioned offset.
    slots = learned_base + gate * gs_offset
    """

    def __init__(self, num_slots: int, slot_dim: int,
                 gs_input_dim: int, gs_hidden_dim: int):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Learned base slots
        self.slot_mu = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.slot_logvar = nn.Parameter(torch.zeros(num_slots, slot_dim) - 2.0)

        # GS encoder
        self.gs_encoder = nn.Sequential(
            nn.Linear(gs_input_dim, gs_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gs_hidden_dim, gs_hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Soft attention pooling
        self.pool_q = nn.Parameter(torch.randn(num_slots, gs_hidden_dim) * 0.02)
        self.pool_k = nn.Linear(gs_hidden_dim, gs_hidden_dim)

        # Centroid → slot offset
        self.centroid_to_offset = nn.Sequential(
            nn.Linear(gs_hidden_dim, gs_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(gs_hidden_dim // 2, slot_dim),
        )

        # Per-slot gate (init negative → sigmoid ≈ 0.12)
        self.gate_param = nn.Parameter(torch.full((num_slots,), -2.0))
        
    def _soft_pool(self, gaussian_features: torch.Tensor) -> torch.Tensor:
        """
        Args:    gaussian_features: [N, C]
        Returns: centroids:         [K, C]
        """
        keys = self.pool_k(gaussian_features)          # [N, C]
        C = keys.shape[-1]

        # [K, C] @ [C, N] → [K, N]
        logits = torch.matmul(self.pool_q, keys.transpose(0, 1)) / (C ** 0.5)
        attn = F.softmax(logits, dim=-1)               # [K, N]

        # [K, N] @ [N, C] → [K, C]
        centroids = torch.matmul(attn, gaussian_features)
        return centroids

    def forward(self, M_0: GSParameter) -> torch.Tensor:
        """
        Args:
            M_0: flat GSParameter [Ng, ...]
                 mu [Ng,3], scale [Ng,3], opacity [Ng,1], sh [Ng,C_sh], cov [Ng,4]
        Returns:
            initial_slots: [K, slot_dim]
        """
        # ── Learned base ──
        if self.training:
            noise = torch.randn_like(self.slot_mu)
            std = torch.exp(0.5 * self.slot_logvar)
            base = self.slot_mu + std * noise       # [K, slot_dim]
        else:
            base = self.slot_mu

        # GS-conditioned offset
        # M_0 is a flat GSParameter [Ng, ...]
        gs_params = torch.cat([
            M_0.mu,                    # [Ng, 3]
            M_0.cov.reshape(M_0.cov.shape[0], -1),    # [Ng, 4]
            M_0.scale,                 # [Ng, 3]
            M_0.opacity,               # [Ng, 1]
            M_0.sh,                    # [Ng, C_sh]
            
        ], dim=-1)                                          # [Ng, gs_input_dim]

        gaussian_features = self.gs_encoder(gs_params)      # [N, gs_hidden_dim]
        centroids = self._soft_pool(gaussian_features)      # [K, gs_hidden_dim]
        offset = self.centroid_to_offset(centroids)         # [K, slot_dim]

        # Gated combination (per-slot gate)
        gate = torch.sigmoid(self.gate_param).unsqueeze(-1) # [K, 1]
        return base + gate * offset                         # [K, slot_dim]

# =============================================================================
# Positional Encoding for Multi-View Features
# =============================================================================
class SpatialViewPositionalEncoding(nn.Module):
    """
    Spatial:  CoordConv (x, y) ∈ [-1, 1], concat + Linear(d+2, d)
    View:     learned per-view embedding, added before concat
    """
    def __init__(self, in_dim: int, max_views: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.max_views = max_views
        self.coord_proj = nn.Linear(in_dim + 2, in_dim, bias=True)
        self.view_pe = nn.Parameter(torch.randn(max_views, 1, in_dim) * 0.02)

        # Cache coords as buffer
        self.register_buffer("_coord_cache", torch.empty(0), persistent=False)
        self._cached_h = -1
        self._cached_w = -1
        
    @staticmethod
    def _coordconv_2d(H: int, W: int, device, dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device, dtype=dtype),
            torch.linspace(-1, 1, W, device=device, dtype=dtype),
            indexing="ij",
        )
        return torch.stack([xx, yy], dim=-1).reshape(H * W, 2)

    def _ensure_cache(self, h: int, w: int, device, dtype):
        if (self._cached_h != h or self._cached_w != w
                or self._coord_cache.numel() == 0
                or str(self._coord_cache.device) != str(device)
                or self._coord_cache.dtype != dtype):
            self._coord_cache = self._coordconv_2d(h, w, device, dtype)
            self._cached_h = h
            self._cached_w = w
        
    def forward(self, feat: torch.Tensor, V: int, Hf: int, Wf: int) -> torch.Tensor:
        """
        Args:
            feat: [Tp, V*Hf*Wf, d] 
            V, Hf, Wf: view count and spatial resolution
        Returns:
            [Tp, V*Hf*Wf, d]
        """
        Tp = feat.shape[0]
        
        # ── Spatial: CoordConv (cached per resolution) ──
        self._ensure_cache(Hf, Wf, feat.device, feat.dtype)  
        coords = self._coord_cache  # [Hf*Wf, 2]
        coords = coords.repeat(V, 1) 
        coords = coords.unsqueeze(0).expand(Tp, -1, -1)         # [Tp, V*Hf*Wf, 2]

        # ── View PE: add per-view embedding ──
        if V > self.max_views:
            raise ValueError(f"Number of views {V} exceeds max_views {self.max_views}")
        feat = feat.view(Tp, V, Hf * Wf, self.in_dim)           # [Tp, V, Hf*Wf, d]
        feat = feat + self.view_pe[:V].unsqueeze(0)              # broadcast [1, V, 1, d]
        feat = feat.reshape(Tp, V * Hf * Wf, self.in_dim)       # [Tp, V*Hf*Wf, d]

        # ── Concat coords + project (single batched Linear call) ──
        feat = torch.cat([feat, coords], dim=-1)                 # [Tp, V*Hf*Wf, d+2]
        feat = self.coord_proj(feat)                             # [Tp, V*Hf*Wf, d]

        return feat

# =============================================================================
# Slot Attention core
# =============================================================================
class SlotAttentionBlock(nn.Module):
    """
    Inputs:
      inputs: [N, slot_dim]
      slots:  [K, slot_dim]  (init or previous frame)
    Returns:
      slots:     [K, slot_dim]
      attn_masks: [K, N]      
    """
    def __init__(self, slot_dim: int, num_iters: int = 3, eps: float = 1e-8, hidden_dim: int = 128, attn_dim: Optional[int] = None):
        super().__init__()
        self.slot_dim = int(slot_dim)
        self.num_iters = int(num_iters)
        self.eps = float(eps)
        self.attn_dim = int(attn_dim) if attn_dim is not None else self.slot_dim
        
        # Normalize inputs and slots
        self.norm_inputs = nn.LayerNorm(self.slot_dim)
        self.norm_slots  = nn.LayerNorm(self.slot_dim)
        self.norm_mlp    = nn.LayerNorm(self.slot_dim)
        
        # Linear projections for attention
        self.to_q = nn.Linear(self.slot_dim, self.attn_dim, bias=False)
        self.to_k = nn.Linear(self.slot_dim,  self.attn_dim, bias=False)
        self.to_v = nn.Linear(self.slot_dim,  self.attn_dim, bias=False)
        
        # Slot update
        self.gru = nn.GRUCell(self.attn_dim, self.slot_dim)
        
        # Slot refinement MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.slot_dim)
        )
        
        self.scale = self.attn_dim ** -0.5

    def forward(self, inputs: torch.Tensor, slots: torch.Tensor, tau: float = 1.0):
        """
        inputs: [N, D]
        slots: [K, D] (K=slot_num, D=slot_dim)
        tau: softmax temperature (1.0 = normal, <1.0 = sharper)
        """

        inputs = self.norm_inputs(inputs)           # [N, slot_dim]
        k = self.to_k(inputs)                       # [N, attn_dim]
        v = self.to_v(inputs)                       # [N, attn_dim]

        for _ in range(self.num_iters):

            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm)               # [slot_num, attn_dim]

            #  Attention logits: [slot_num, attn_dim] @ [attn_dim, N] → [slot_num, N]
            attn_logits = torch.einsum("kd,nd->kn", q, k) * self.scale / tau

            # (A) Competitive: softmax over slot dim (dim=0)
            attn = F.softmax(attn_logits, dim=0) + self.eps # [slot_num, N], sum over K = 1
            attn_norm = attn / attn.sum(dim=-1, keepdim=True)  # [slot_num, N] normalized over tokens for each slot

            # (B) Weighted aggregation: [slot_num, N] @ [N, slot_dim] → [slot_num, slot_dim]
            updates = torch.einsum("kn,nd->kd", attn_norm, v)  # [slot_num, attn_dim]

            # (C) GRU update + residual MLP
            slots = self.gru(updates, slots)  # [slot_num, slot_dim]
            slots = slots + self.mlp(self.norm_mlp(slots))

        q_final = self.to_q(self.norm_slots(slots))
        dots_final = torch.einsum("kd,nd->kn", q_final, k) * self.scale / tau
        attn_masks = dots_final.softmax(dim=0)  # [slot_num, N]

        return slots, attn_masks

# =============================================================================
# MultiViewTemporalSlotAttention
# =============================================================================

class MultiViewTemporalSlotAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gs_state_dim: int,
        slot_dim: int,
        num_slots: int,
        num_iters: int ,
        hidden_dim: int,
        max_views: int = 8,
        gs_hidden_dim: int = 128,
        eps: float = 1e-8
    ):
        super().__init__()

        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        
        # 1x1 conv to project CNN features → slot_dim: [B*V*Tp, C_out, Hf, Wf] → [B*V*Tp, slot_dim, Hf, Wf]
        self.input_proj = nn.Conv2d(in_dim, self.slot_dim, kernel_size=1, bias=True)  

        # ── Positional encoding ──
        self.positional_encoding = SpatialViewPositionalEncoding(
            in_dim=self.slot_dim,
            max_views=max_views
        )
        
        # ── Slot Attention iteration ──
        self.slot_attention = SlotAttentionBlock(
            slot_dim=self.slot_dim,
            num_iters=num_iters,
            eps=eps,
            hidden_dim=hidden_dim)

        # ── GS-conditioned slot initialization ──
        self.slot_init = SlotInit(
            num_slots=num_slots,
            slot_dim=slot_dim,
            gs_input_dim=gs_state_dim,  # mu[3] + scale[3] + opacity[1] + sh[C_sh] + cov[4]
            gs_hidden_dim=gs_hidden_dim
        )
    
    
    def forward_perbatch(self, feats: torch.Tensor, gs_state: GSParameter, tau: float = 1.0):
        """
        feat: [V, Tp, C_out, Hf, Wf]
        gs_state: flat GSParameter [Ng, ...]
        returns:
            slots: [Tp, K, d]
            attn_masks: [Tp, K, V*Hf*Wf]
        """
        V, Tp, C_out, Hf, Wf = feats.shape
        # Project: [V*T, C_in, Hf, Wf] → [V*T, slot_dim, Hf, Wf]
        x = feats.reshape(V * Tp, C_out, Hf, Wf)
        x = self.input_proj(x)                             # [V*T, D, Hf, Wf]
        D = x.shape[1]

        # Reshape for positional encoding: [V*T, D, Hf, Wf] → [Tp, V*Hf*Wf, D]
        x = x.view(V, Tp, D, Hf, Wf).permute(1, 0, 3, 4, 2).contiguous()  # [Tp, V, Hf, Wf, D]
        x = x.reshape(Tp, V * Hf * Wf, D)                  # [Tp, V*Hf*Wf, D]

        # Initialize slots from gs_state
        slots = self.slot_init(gs_state)                    # [K, slot_dim]

        # Temporal loop with carry-over
        all_slots = []
        all_attn = []

        x_pe = self.positional_encoding(x, V, Hf, Wf)

        for t in range(Tp):
            slots, attn = self.slot_attention(x_pe[t], slots, tau=tau)
            all_slots.append(slots)   # [K, slot_dim]
            all_attn.append(attn)     # [K, V*Hf*Wf]

        return torch.stack(all_slots, dim=0), torch.stack(all_attn, dim=0)

    def forward(
        self,
        features: torch.Tensor,        # [B, V, Tp, C_out, Hf, Wf]
        gs_states: List[GSParameter],   # len B, each flat [Ng, ...]
        tau: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = features.shape[0]

        all_slots = []
        all_attn = []

        for b in range(B):
            slots_b, attn_b = self.forward_perbatch(features[b], gs_states[b], tau=tau)
            all_slots.append(slots_b)
            all_attn.append(attn_b)

        return torch.stack(all_slots, dim=0), torch.stack(all_attn, dim=0)