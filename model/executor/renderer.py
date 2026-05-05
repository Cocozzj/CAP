"""3DGS rendering for reconstruction loss.

Wraps ``gsplat.rasterization`` so the Executor can produce ``rendered_frames``
that the rec / lpips / depth losses compare against the GT video.

Lazy import: if gsplat isn't installed we return None and the rec losses
gracefully fall back to 0 (matches existing ``reconstruction_loss`` early exit).

Design choices for training-time cost:
  - Render only a SUBSET of timesteps per training step (default: initial +
    final).  Rendering all T=30 frames × V=3 views × B=8 = 720 rasterizations
    per step is too slow.  Two timesteps (96 rasterizations) gives enough
    signal to learn coherent Gaussians + final-state dynamics.
  - Use ``covars=`` directly with SceneState's full 3×3 covariance — saves
    the quat→rot decomposition that the (quat, scale) parameterisation needs.
  - SH degree 0 (DC term only).  The dataloader stores f_dc_0/1/2 at sh[:, 0:3];
    higher SH bands use the Inria-3DGS permuted layout which we can wire
    later if we need view-dependent appearance for rec_loss.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

try:
    from gsplat import rasterization                           # type: ignore
    _GSPLAT_OK = True
except ImportError:                                             # pragma: no cover
    _GSPLAT_OK = False
    rasterization = None  # type: ignore


def gsplat_available() -> bool:
    """True iff gsplat can be imported.  Caller skips rendering if False."""
    return _GSPLAT_OK


# ════════════════════════════════════════════════════════════════════
# Single-scene render
# ════════════════════════════════════════════════════════════════════

def _flatten_scene_for_sample(
    scene, b: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten one batch sample's [K, N_max, ...] padded Gaussians to [N_valid, ...]
    by indexing with ``scene.mask``.  Returns (mu, cov, sh, opacity)."""
    mask_b = scene.mask[b]                                      # [K, N_max] bool
    mu_b   = scene.mu[b][mask_b]                                # [N_valid, 3]
    cov_b  = scene.cov[b][mask_b]                               # [N_valid, 3, 3]
    sh_b   = scene.sh[b][mask_b]                                # [N_valid, C_sh]
    op_b   = scene.opacity[b][mask_b].squeeze(-1)               # [N_valid]
    return mu_b, cov_b, sh_b, op_b


def _sh_to_colors(sh: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Convert dataloader's flat [N, c_sh] SH layout → gsplat's [N, K, 3] colors.

    Dataloader stores 3DGS .ply attributes verbatim (Inria convention):
        sh[:, 0:3]   = f_dc_0/1/2          (DC coeffs, interleaved RGB)
        sh[:, 3:18]  = f_rest_0..14        (R coefficients of bands 1, 2, 3)
        sh[:, 18:33] = f_rest_15..29       (G coefficients of bands 1, 2, 3)
        sh[:, 33:48] = f_rest_30..44       (B coefficients of bands 1, 2, 3)

    gsplat wants [N, K, 3] where K=(degree+1)^2, INTERLEAVED RGB per band index.
    For degree 3 → K=16:
        out[:, 0,  :] = DC RGB
        out[:, k,  c] = bands-1-3 coefficient (k-1) of channel c, for k in 1..15
    """
    if sh.shape[-1] >= 48:
        # SH degree 3 — full Inria layout
        dc     = sh[:, :3]                                    # [N, 3]
        rest_R = sh[:, 3:18]                                  # [N, 15]
        rest_G = sh[:, 18:33]                                 # [N, 15]
        rest_B = sh[:, 33:48]                                 # [N, 15]
        rest   = torch.stack([rest_R, rest_G, rest_B], dim=-1)  # [N, 15, 3]
        colors = torch.cat([dc.unsqueeze(1), rest], dim=1)    # [N, 16, 3]
        return colors, 3
    if sh.shape[-1] >= 3:
        # Fallback: SH degree 0 (DC only) when fewer coefficients are present
        return sh[:, :3].unsqueeze(1), 0
    raise ValueError(f"sh has too few coefficients: {sh.shape[-1]}")


def render_scene(
    scene,                                                       # SceneState
    intrinsics: torch.Tensor,                                    # [B, V, 3, 3]
    extrinsics: torch.Tensor,                                    # [B, V, 4, 4] world→cam
    image_size: Tuple[int, int] = (256, 256),
    render_depth: bool = True,
) -> Optional[dict]:
    """Render the scene from V cameras for each of B batch samples.

    Returns:
        {
          "rgb":   [B, V, 3, H, W],
          "depth": [B, V, 1, H, W]  (or None if ``render_depth=False``),
          "alpha": [B, V, 1, H, W],
        }
        or None if gsplat is not installed.

    Renders sample-by-sample because N_valid varies per batch.
    """
    if not _GSPLAT_OK:
        return None

    B = scene.B
    V = intrinsics.shape[1]
    H, W = image_size
    device = scene.mu.device

    rgb_all:   List[torch.Tensor] = []
    depth_all: List[torch.Tensor] = []
    alpha_all: List[torch.Tensor] = []

    mode = "RGB+ED" if render_depth else "RGB"

    for b in range(B):
        mu_b, cov_b, sh_b, op_b = _flatten_scene_for_sample(scene, b)

        if mu_b.numel() == 0:
            rgb_all.append(torch.zeros(V, 3, H, W, device=device, dtype=torch.float32))
            if render_depth:
                depth_all.append(torch.zeros(V, 1, H, W, device=device, dtype=torch.float32))
            alpha_all.append(torch.zeros(V, 1, H, W, device=device, dtype=torch.float32))
            continue

        colors, sh_degree = _sh_to_colors(sh_b)
        N = mu_b.size(0)

        # gsplat expects fp32 throughout; AMP is disabled for the rasterizer.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            renders, alphas, _ = rasterization(
                means     = mu_b.float(),
                # When ``covars`` is given, gsplat ignores quats/scales — but it
                # still requires shape-compatible placeholders to type-check.
                quats     = torch.zeros(N, 4, device=device, dtype=torch.float32),
                scales    = torch.ones(N, 3, device=device, dtype=torch.float32),
                opacities = op_b.float(),
                colors    = colors.float(),
                viewmats  = extrinsics[b].float(),
                Ks        = intrinsics[b].float(),
                width     = W,
                height    = H,
                sh_degree = sh_degree,
                render_mode = mode,
                covars    = cov_b.float(),
            )
        # renders: [V, H, W, 3 or 4]   alphas: [V, H, W, 1]
        if render_depth:
            rgb_b   = renders[..., :3]
            depth_b = renders[..., 3:4]
        else:
            rgb_b = renders
            depth_b = None

        rgb_all.append(rgb_b.permute(0, 3, 1, 2).contiguous())   # [V, 3, H, W]
        if render_depth:
            depth_all.append(depth_b.permute(0, 3, 1, 2).contiguous())
        alpha_all.append(alphas.permute(0, 3, 1, 2).contiguous())

    return {
        "rgb":   torch.stack(rgb_all,   dim=0),                  # [B, V, 3, H, W]
        "depth": torch.stack(depth_all, dim=0) if render_depth else None,
        "alpha": torch.stack(alpha_all, dim=0),
    }


# ════════════════════════════════════════════════════════════════════
# Trajectory render — renders a sparse subset of timesteps
# ════════════════════════════════════════════════════════════════════

def _uniform_indices(T: int, n: int) -> List[int]:
    """Uniformly spaced int indices over [0, T] inclusive, length ≤ n.

    Examples (T=29):
        n=2 → [0, 29]
        n=5 → [0, 7, 14, 21, 29]
        n=1 → [29]   (final only)
    Duplicates are removed so callers always get distinct timesteps.
    """
    if T <= 0:
        return [0]
    if n <= 1:
        return [T]
    step = T / (n - 1)
    return sorted({int(round(i * step)) for i in range(n)})


def render_trajectory(
    initial_scene,                                               # SceneState
    trajectory: List,                                            # List[SceneState] length T
    intrinsics: torch.Tensor,                                    # [B, V, 3, 3]
    extrinsics: torch.Tensor,                                    # [B, V, 4, 4]
    image_size: Tuple[int, int] = (256, 256),
    render_depth: bool = True,
    timestep_indices: Optional[List[int]] = None,
    n_timesteps: Optional[int] = None,
) -> Optional[dict]:
    """Render a trajectory of SceneStates.

    Args:
        initial_scene:    pre-execution scene (= timestep 0)
        trajectory:       list of T post-step SceneStates (timesteps 1..T)
        timestep_indices: explicit indices to render (takes priority).
        n_timesteps:      if no explicit indices, render this many uniformly
            spaced steps over [0, T].  ``n=0`` returns None (skip rendering).
            Defaults to 2 (initial + final) if both are None.

    Returns:
        {
          "rgb":               [B, V, T_rendered, 3, H, W],
          "depth":             [B, V, T_rendered, 1, H, W]  (or None),
          "alpha":             [B, V, T_rendered, 1, H, W],
          "timestep_indices":  list[int]  the actual indices rendered (for GT subsampling),
          "T_total":           int        len(trajectory) — used by loss to align GT
        }
        or None if gsplat unavailable, or if ``n_timesteps == 0``.
    """
    if not _GSPLAT_OK:
        return None
    if n_timesteps is not None and n_timesteps == 0:
        return None  # explicit "skip render" signal

    T = len(trajectory)
    if timestep_indices is None:
        n = n_timesteps if n_timesteps is not None else 2
        timestep_indices = _uniform_indices(T, n)

    rgbs:   List[torch.Tensor] = []
    depths: List[torch.Tensor] = []
    alphas: List[torch.Tensor] = []

    for t_idx in timestep_indices:
        if t_idx == 0:
            scene_t = initial_scene
        else:
            # trajectory is 1-indexed in semantics: trajectory[t-1] = post-step-t
            scene_t = trajectory[min(t_idx - 1, T - 1)]
        out = render_scene(scene_t, intrinsics, extrinsics, image_size, render_depth)
        if out is None:
            return None
        rgbs.append(out["rgb"])                                  # [B, V, 3, H, W]
        if render_depth:
            depths.append(out["depth"])
        alphas.append(out["alpha"])

    return {
        "rgb":              torch.stack(rgbs,   dim=2),          # [B, V, T_rendered, 3, H, W]
        "depth":            torch.stack(depths, dim=2) if render_depth and depths else None,
        "alpha":            torch.stack(alphas, dim=2),
        "timestep_indices": list(timestep_indices),
        "T_total":          T,
    }


__all__ = [
    "gsplat_available",
    "render_scene",
    "render_trajectory",
]
