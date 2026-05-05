"""
deform/fallback.py — Black-box appearance update when physics is disabled.

Per PDF §1.1: when ``enable_physics=False`` (e.g. Stage-0 RIGID training
curriculum, or in lightweight inference), the deformation branch reduces to
a learned Δscale + Δopacity update.  No position / covariance change.

This is a deliberate degenerate path — physics is the principled extension,
fallback is the "always-runs" minimal version.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BlackBoxFallback(nn.Module):
    """Learned (Δscale, Δopacity) from ρ when physics is off.

    Outputs are scaled to small magnitudes so that the no-physics branch
    can't dominate over the rigid SE(3) update.
    """

    def __init__(self, rho_dim: int = 16) -> None:
        super().__init__()
        # ρ → (Δscale[3], Δopacity[1])
        self.head = nn.Sequential(
            nn.Linear(rho_dim, 32), nn.GELU(),
            nn.Linear(32, 4),
        )

    def forward(self, rho: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rho: [B, K, rho_dim]

        Returns:
            delta_scale:   [B, K, 3]    log-scale increment (small)
            delta_opacity: [B, K, 1]    opacity increment (small)
        """
        # NaN guard — rho can come from upstream physics propagation.  The
        # rho_parser path NaN-cleans inside RhoParser, but the fallback path
        # bypasses it, so we have to clean here.
        rho = torch.nan_to_num(rho, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.head(rho)
        # Small magnitude so that fallback acts as a fine appearance correction
        delta_scale   = out[..., :3] * 0.1
        delta_opacity = out[..., 3:] * 0.05
        return delta_scale, delta_opacity
