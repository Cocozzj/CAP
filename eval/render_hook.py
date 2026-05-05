"""Optional 3DGS render adapter.

PDF only abstractly defines a renderer R: Gaussians → images / point clouds
without naming a specific library.  This module tries the popular options
(gsplat, nerfacc) lazily; if none are installed, returns ``None`` and lets
the caller skip image-level metrics.

Usage::
    from .render_hook import render_scene, available_backend
    print("Renderer:", available_backend())   # "gsplat" | "nerfacc" | None
    images = render_scene(scene, camera_params)   # [V, 3, H, W] or None
"""

from __future__ import annotations

import functools
import warnings
from typing import Optional

import torch


@functools.lru_cache(maxsize=1)
def available_backend() -> Optional[str]:
    """Return the name of the first installable 3DGS renderer found, or None."""
    for name in ("gsplat", "nerfacc"):
        try:
            __import__(name)
            return name
        except ImportError:
            continue
    return None


def render_scene(
    scene,
    camera_params: dict,
    image_size: tuple = (256, 256),
) -> Optional[torch.Tensor]:
    """Render a SceneState through the available 3DGS backend.

    Args:
        scene:         SceneState  (mu / cov / sh / opacity)
        camera_params: dict with intrinsics + extrinsics (per-renderer format)
        image_size:    (H, W)

    Returns:
        rendered: [V, 3, H, W] in [0, 1], or ``None`` if no renderer installed.

    Note: the actual renderer wiring is left as TODO since each library
    expects different camera conventions and sh-degree assumptions.
    Until you decide on a renderer, this returns None and the caller
    should skip PSNR/LPIPS computation.
    """
    backend = available_backend()
    if backend is None:
        warnings.warn(
            "No 3DGS renderer installed (tried gsplat, nerfacc). "
            "Skipping image-level metrics.  Install gsplat to enable: "
            "  pip install gsplat",
            stacklevel=2,
        )
        return None

    # TODO: implement per-backend rendering when renderer is finalised.
    warnings.warn(
        f"render_scene: {backend!r} is installed but the integration is TODO. "
        "Implement gsplat/nerfacc rasterization here when renderer choice "
        "is finalised.",
        stacklevel=2,
    )
    return None
