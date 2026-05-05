"""Reconstruction + scene-distance metrics shared across eval scripts."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from model.utils import SceneState, masked_mean


# ──────────────────────────────────────────────────────────────────────
# Image metrics
# ──────────────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, gt: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """PSNR between two image tensors in [0, max_val].

    Inputs may be any shape; reduces over all dims.
    """
    mse = F.mse_loss(pred, gt)
    if mse.item() < 1e-12:
        return torch.tensor(float("inf"), device=mse.device)
    return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=mse.device) / mse)


def lpips_score(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """LPIPS perceptual distance.  Lazy-loads ``lpips``; returns 0 if unavailable.

    Inputs are flattened to [N, 3, H, W] and rescaled to [-1, 1].
    """
    try:
        import lpips
    except ImportError:
        return pred.new_zeros(())
    if not hasattr(lpips_score, "_net"):
        lpips_score._net = lpips.LPIPS(net="alex").to(pred.device)
    p = (pred.flatten(0, -4) * 2 - 1).clamp(-1, 1)
    g = (gt.flatten(0, -4) * 2 - 1).clamp(-1, 1)
    return lpips_score._net(p, g).mean()


# ──────────────────────────────────────────────────────────────────────
# Scene distance
# ──────────────────────────────────────────────────────────────────────

def scene_distance_metric(
    pred_state: SceneState,
    gt_state:   SceneState,
) -> torch.Tensor:
    """Mean L2 displacement on Gaussian centres, mask-aware (eval version)."""
    diff = (pred_state.mu - gt_state.mu).norm(dim=-1)               # [B, K, N]
    if pred_state.mask is not None:
        return masked_mean(diff, pred_state.mask, dim=-1).mean()
    return diff.mean()


# ──────────────────────────────────────────────────────────────────────
# Trajectory comparison
# ──────────────────────────────────────────────────────────────────────

def trajectory_distance(
    pred_traj: list,                 # List[SceneState]
    gt_traj:   list,
) -> torch.Tensor:
    """Mean per-step scene distance over an aligned trajectory pair."""
    assert len(pred_traj) == len(gt_traj), "trajectory lengths differ"
    if not pred_traj:
        return torch.zeros(())
    dists = [scene_distance_metric(p, g) for p, g in zip(pred_traj, gt_traj)]
    return torch.stack(dists).mean()
