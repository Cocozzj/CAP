"""
Motion Encoder (Stage 2).

    slots [B, Tp, K, d_slot]  +  phi (CanonicalFrame)  +  optional mask [B, Tp, K]
    → FiLM conditioning (canonical-aware modulation)
    → TCN backbone (dilated, weight-normed, optional causal)
    → Temporal windowing (masked average pool → T_act)
    → motion_features [B, T_act, K, motion_dim]
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# FiLM: Feature-wise Linear Modulation conditioned on canonical frame
# ===========================================================================
class CanonicalFiLM(nn.Module):
    """
    Modulate slot features based on canonical frame phi (CanonicalFrame).

    phi → MLP → (gamma, beta)
    output = gamma * slots + beta

    This "rotates" slot features into canonical space via learned modulation,
    making downstream motion encoding pose-invariant.
    """

    def __init__(self, in_dim: int, phi_dim: int = 12, hidden_dim: int = 64):
        """
        Args:
            in_dim:   dimension of input slot features
            phi_dim:    R_flat(9) + t(3) = 12
            hidden_dim: MLP hidden size
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(phi_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.to_gamma = nn.Linear(hidden_dim, in_dim)
        self.to_beta = nn.Linear(hidden_dim, in_dim)

        # Init: gamma ≈ 1, beta ≈ 0 → identity modulation at start
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.ones_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(
        self,
        slots: torch.Tensor,       # [B, Tp, K, d_slot]
        R: torch.Tensor,           # [B, K, 3, 3]
        t: torch.Tensor,           # [B, K, 3]
    ) -> torch.Tensor:
        """
        Returns:
            conditioned_slots: [B, Tp, K, d_slot]
        """
        B, K = R.shape[0], R.shape[1]

        # ── Build phi vector: [B, K, 12] ──
        R_flat = R.reshape(B, K, 9)                             # [B, K, 9]
        phi = torch.cat([R_flat, t], dim=-1)                    # [B, K, 12]

        # ── MLP → gamma, beta: [B, K, d_slot] ──
        h = self.mlp(phi)                                       # [B, K, hidden]
        gamma = self.to_gamma(h)                                # [B, K, d_slot]
        beta = self.to_beta(h)                                  # [B, K, d_slot]

        # ── Broadcast over Tp and modulate ──
        gamma = gamma.unsqueeze(1)                              # [B, 1, K, d_slot]
        beta = beta.unsqueeze(1)                                # [B, 1, K, d_slot]

        return gamma * slots + beta                             # [B, Tp, K, d_slot]


# ===========================================================================
# TCN: Temporal Convolutional Network (Bai et al. 2018 style)
# ===========================================================================
class Chomp1d(nn.Module):
    """Remove trailing padding for causal convolution."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.chomp_size == 0 else x[..., :-self.chomp_size]


class _TCNBlock(nn.Module):
    """
    Single TCN residual block: two weight-normed dilated convolutions
    with optional causal masking via Chomp1d.

    Input/output: [N, C, L]
    """
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float, causal: bool):
        super().__init__()
        if causal:
            pad = (kernel_size - 1) * dilation
            chomp = pad
        else:
            pad = ((kernel_size - 1) * dilation) // 2
            chomp = 0

        conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=pad, dilation=dilation)
        nn.init.kaiming_normal_(conv1.weight, nonlinearity="relu")
        self.conv1 = nn.utils.parametrizations.weight_norm(conv1)
        self.chomp1 = Chomp1d(chomp)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                          padding=pad, dilation=dilation)
        nn.init.kaiming_normal_(conv2.weight, nonlinearity="relu")
        self.conv2 = nn.utils.parametrizations.weight_norm(conv2)
        self.chomp2 = Chomp1d(chomp)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = None
        if n_inputs != n_outputs:
            ds = nn.Conv1d(n_inputs, n_outputs, 1)
            nn.init.kaiming_normal_(ds.weight, nonlinearity="linear")
            self.downsample = ds
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.relu2(self.chomp2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    """
    Multi-layer dilated TCN with residual connections.

    Each layer doubles dilation: 1, 2, 4, 8, ...
    Two convs per block with weight normalization and dropout.
    Supports causal mode (for autoregressive use) via Chomp1d.
    """
    def __init__(self, in_channels: int, channels: Sequence[int],
                 kernel_size: int = 3, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else channels[i - 1]
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size,
                                    dilation, dropout, causal))
        self.network = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, C_in, L] → [N, C_out, L]"""
        return self.network(x)


# ===========================================================================
# MotionEncoder: FiLM + TCN + masked temporal windowing
# ===========================================================================
class MotionEncoder(nn.Module):
    """
    Stage 2: Extract per-object motion features from temporal slot representations.

    Pipeline:
        1. FiLM conditioning: modulate slots using canonical frame phi
        2. TCN: extract temporal motion patterns per object
        3. Temporal windowing: masked average-pool over fixed windows → T_act tokens

    Input:
        slots:  [B, Tp, K, d_slot]
        phi:    CanonicalFrame (R_w2c: [B, K, 3, 3], t_w2c: [B, K, 3])
        mask:   [B, Tp, K] optional — 1=visible, 0=occluded (None = all visible)

    Output:
        motion_features: [B, T_act, K, motion_dim]
        T_act = Tp // window_size  (padded if not divisible)
    """

    def __init__(self, motion_param: dict):
        super().__init__()

        in_dim = int(motion_param.get("in_dim", 128))
        motion_dim = int(motion_param.get("motion_dim", 128))
        window_size = int(motion_param.get("window_size", 10))
        dropout = float(motion_param.get("dropout", 0.1))
        causal = bool(motion_param.get("causal", False))
        kernel_size = int(motion_param.get("kernel_size", 3))
        channels = list(motion_param.get("channels", [motion_dim] * 4))
        film_hidden = int(motion_param.get("film_hidden", 64))

        self.in_dim = in_dim
        self.motion_dim = motion_dim
        self.window_size = window_size

        # ── FiLM conditioning ──
        self.film = CanonicalFiLM(
            in_dim=in_dim,
            phi_dim=12,
            hidden_dim=film_hidden,
        )

        # ── TCN backbone ──
        self.tcn = TCNBackbone(
            in_channels=in_dim,
            channels=channels,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=causal,
        )
        assert self.tcn.out_channels == motion_dim, (
            f"TCN last channel ({self.tcn.out_channels}) must equal "
            f"motion_dim ({motion_dim}). Check 'channels' config."
        )

    def _temporal_window(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        min_visible_ratio: float = 0.3,
    ) -> tuple:
        """
        Non-overlapping masked average pooling over time.

        When mask is None (Dataset-A, full visibility):
            plain AvgPool1d — equivalent to all 1s mask.

        When mask is provided (Dataset-B, possible occlusion):
            weighted average, occluded frames excluded.

        Args:
            x:    [N, C, Tp]
            mask: [N, 1, Tp] or None  — 1.0 = visible, 0.0 = occluded
            min_visible_ratio: fraction of visible frames needed for a
                               window to be considered valid (for seq_mask)
        Returns:
            pooled:      [N, C, T_act]     — pooled features
            window_mask: [N, T_act] bool   — True = window has enough visible frames
                         None if mask is None (all valid)
        """
        Tp = x.shape[-1]
        w = self.window_size

        # ── Pad if Tp not divisible by window_size ──
        remainder = Tp % w
        if remainder != 0:
            pad_size = w - remainder
            x = F.pad(x, (0, pad_size), mode="replicate")
            if mask is not None:
                # Padded frames are NOT visible
                mask = F.pad(mask, (0, pad_size), value=0.0)

        if mask is None:
            # ── Fast path: plain AvgPool (Dataset-A) ──
            return F.avg_pool1d(x, kernel_size=w, stride=w), None
        else:
            # ── Masked path: weighted average (Dataset-B) ──
            N, C, L = x.shape
            T_act = L // w
            x_win = x.reshape(N, C, T_act, w)                  # [N, C, T_act, w]
            m_win = mask.reshape(N, 1, T_act, w)                # [N, 1, T_act, w]

            # Weighted sum / count
            count = m_win.sum(dim=-1).clamp_min(1.0)            # [N, 1, T_act]
            pooled = (x_win * m_win).sum(dim=-1) / count        # [N, C, T_act]

            # ── Window validity: enough visible frames? ──
            visible_ratio = m_win.squeeze(1).mean(dim=-1)       # [N, T_act]
            window_mask = (visible_ratio >= min_visible_ratio)   # [N, T_act] bool

            return pooled, window_mask

    def forward(
        self,
        slots: torch.Tensor,                   # [B, Tp, K, d_slot]
        phi,                                    # CanonicalFrame
        mask: Optional[torch.Tensor] = None,    # [B, Tp, K] — 1=visible, 0=occluded
    ) -> Dict[str, Any]:
        """
        Args:
            slots:  [B, Tp, K, d_slot]
            phi:    CanonicalFrame (R_w2c: [B, K, 3, 3], t_w2c: [B, K, 3])
            mask:   [B, Tp, K] optional visibility mask
                    None = all visible (Dataset-A fast path)
                    float tensor = per-frame per-object visibility (Dataset-B)

        Returns:
            {
                "z_motion":    [B, T_act, K, motion_dim],
                "window_mask": [B, T_act, K] bool | None,
            }
        """
        B, Tp, K, D = slots.shape
        R = phi.R_w2c                                               # [B, K, 3, 3]
        t = phi.t_w2c                                               # [B, K, 3]

        # ── 1. FiLM conditioning ──
        conditioned = self.film(slots, R, t)                    # [B, Tp, K, d_slot]

        # ── 2. TCN per object ──
        # Reshape: [B, Tp, K, D] → [B*K, D, Tp]  (channels-first for Conv1d)
        x = conditioned.permute(0, 2, 3, 1).contiguous()       # [B, K, D, Tp]
        x = x.reshape(B * K, D, Tp)                            # [B*K, D, Tp]

        x = self.tcn(x)                                         # [B*K, motion_dim, Tp]

        # ── 3. Prepare mask for windowing ──
        tcn_mask = None
        if mask is not None:
            # [B, Tp, K] → [B, K, Tp] → [B*K, 1, Tp]
            tcn_mask = mask.permute(0, 2, 1).contiguous()       # [B, K, Tp]
            tcn_mask = tcn_mask.reshape(B * K, 1, Tp)           # [B*K, 1, Tp]

        # ── 4. Temporal windowing (masked or plain) ──
        x, window_mask_flat = self._temporal_window(x, tcn_mask)  # [B*K, motion_dim, T_act], [B*K, T_act]|None

        # ── Reshape back ──
        T_act = x.shape[-1]
        motion_dim = x.shape[1]
        x = x.reshape(B, K, motion_dim, T_act)                   # [B, K, motion_dim, T_act]
        motion_features = x.permute(0, 3, 1, 2).contiguous()   # [B, T_act, K, motion_dim]

        # ── Window mask: [B*K, T_act] → [B, T_act, K] ──
        if window_mask_flat is not None:
            window_mask = window_mask_flat.reshape(B, K, T_act)   # [B, K, T_act]
            window_mask = window_mask.permute(0, 2, 1).contiguous()  # [B, T_act, K]
        else:
            window_mask = None

        return {
            "z_motion": motion_features,                        # [B, T_act, K, motion_dim]
            "window_mask": window_mask,                         # [B, T_act, K] bool | None
        }