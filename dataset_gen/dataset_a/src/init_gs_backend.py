"""3DGS backends for Step 4 (init_gs from first frame).

Two implementations behind a single GSBackend interface:

  - GsplatBackend:   per-scene optimization with random init + L1 loss.
                     ~3-5 sec / trajectory at 3000 iter.
                     Quality: blob-like with only 3 views (depth ambiguity).
                     Use as fallback only.

  - MVSplatBackend:  feed-forward 3DGS from posed multi-view images.
                     Requires installing MVSplat + writing
                     `src/mvsplat_adapter.py`. Currently not implemented.

The PRIMARY backend is `mesh` (in `init_gs_mesh.py`), which directly samples
PartNet's mesh files at the trajectory's first joint state. That gives clean,
correct geometry without any optimization. Use `--backend mesh` for production.

All return a dict:
    {
      "means":      (N, 3) float32,
      "scales":     (N, 3) float32  (linear),
      "quats":      (N, 4) float32  (wxyz),
      "opacities":  (N,)   float32  in [0,1],
      "colors":     (N, 3) float32  in [0,1],
    }
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# common interface
# ----------------------------------------------------------------------------
class GSBackend(ABC):
    @abstractmethod
    def reconstruct(
        self,
        rgb_views: List[np.ndarray],         # (H, W, 3) uint8 or float per view
        intrinsics: List[np.ndarray],        # (3, 3) per view
        c2w: List[np.ndarray],               # (4, 4) camera-to-world per view
    ) -> Optional[dict]:
        """Returns the GS dict, or None on failure."""
        ...


# ============================================================================
# MVSplat backend (not yet wired up — adapter file required)
# ============================================================================
class MVSplatBackend(GSBackend):
    """Feed-forward 3DGS from MVSplat. Requires `src/mvsplat_adapter.py` that
    exposes `load_model(checkpoint, device)` and
    `infer(model, rgbs, Ks, c2w) -> gs_dict`."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from . import mvsplat_adapter  # type: ignore[import]
        except ImportError as e:
            raise RuntimeError(
                "MVSplat adapter missing. Install MVSplat and create "
                "`dataset/dataset_a/src/mvsplat_adapter.py` with "
                "load_model() and infer()."
            ) from e
        self._model = mvsplat_adapter.load_model(self.checkpoint_path, self.device)

    def reconstruct(self, rgb_views, intrinsics, c2w):
        from . import mvsplat_adapter  # type: ignore[import]
        self._load()
        try:
            return mvsplat_adapter.infer(self._model, rgb_views, intrinsics, c2w)
        except Exception as e:  # noqa: BLE001
            logger.exception("MVSplat inference failed: %s", e)
            return None


# ============================================================================
# gsplat backend (per-scene optimization, fallback only)
# ============================================================================
class GsplatBackend(GSBackend):
    """Per-scene 3DGS optimization on the first frame's 3 views.

    Random init in a small bounding box, plain L1 loss vs all 3 views.
    With only 3 views and no densify/prune, quality is limited (PSNR ~22-25).
    Use `mesh` backend for clean geometry; this is here as a fallback.
    """

    def __init__(self, iters: int = 3000, init_n_points: int = 50_000,
                 device: str = "cuda"):
        self.iters = iters
        self.init_n_points = init_n_points
        self.device = device

    def reconstruct(self, rgb_views, intrinsics, c2w):
        try:
            import torch
            import torch.nn.functional as F
            from gsplat import rasterization
        except ImportError as e:
            logger.error("gsplat not installed: %s", e)
            return None

        device = self.device

        # Stack views into a tensor
        rgbs = []
        for img in rgb_views:
            arr = np.asarray(img)
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            elif arr.max() > 1.5:
                arr = arr.astype(np.float32) / 255.0
            rgbs.append(arr)
        rgbs = torch.from_numpy(np.stack(rgbs)).float().to(device)
        N, H, W, _ = rgbs.shape
        Ks = torch.from_numpy(np.stack(intrinsics)).float().to(device)
        c2ws = torch.from_numpy(np.stack(c2w)).float().to(device)

        # Random init in 1.5m box around scene origin
        pts = (torch.rand(self.init_n_points, 3, device=device) - 0.5) * 1.5
        cols = torch.rand(self.init_n_points, 3, device=device) * 0.5 + 0.25

        means = torch.nn.Parameter(pts)
        scales_log = torch.nn.Parameter(torch.full((pts.shape[0], 3), -3.0, device=device))
        quats = torch.nn.Parameter(
            torch.tensor([1, 0, 0, 0], device=device, dtype=torch.float32)
                 .repeat(pts.shape[0], 1)
        )
        opacity_lgt = torch.nn.Parameter(torch.full((pts.shape[0],), 0.5, device=device))
        colors = torch.nn.Parameter(cols)

        optim = torch.optim.Adam([
            {"params": [means],       "lr": 1.6e-3},
            {"params": [scales_log],  "lr": 5e-3},
            {"params": [quats],       "lr": 1e-3},
            {"params": [opacity_lgt], "lr": 5e-2},
            {"params": [colors],      "lr": 2.5e-3},
        ])

        for it in range(self.iters):
            v = torch.randint(0, N, (1,)).item()
            viewmat = torch.linalg.inv(c2ws[v])
            try:
                rendered, _, _ = rasterization(
                    means=means,
                    quats=F.normalize(quats, dim=-1),
                    scales=scales_log.exp(),
                    opacities=torch.sigmoid(opacity_lgt),
                    colors=colors,
                    viewmats=viewmat[None],
                    Ks=Ks[v][None],
                    width=W, height=H,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("gsplat rasterization failed at iter %d: %s", it, e)
                return None
            loss = F.l1_loss(rendered[0], rgbs[v])
            optim.zero_grad()
            loss.backward()
            optim.step()

        return {
            "means":     means.detach().cpu().numpy().astype(np.float32),
            "scales":    scales_log.exp().detach().cpu().numpy().astype(np.float32),
            "quats":     F.normalize(quats, dim=-1).detach().cpu().numpy().astype(np.float32),
            "opacities": torch.sigmoid(opacity_lgt).detach().cpu().numpy().astype(np.float32),
            "colors":    colors.detach().cpu().numpy().astype(np.float32),
        }


# ============================================================================
# factory
# ============================================================================
def make_backend(gs_cfg: dict, device: str = "cuda") -> GSBackend:
    """Construct a GSBackend from the `gs:` block of default.yaml.

    Note: backend 'mesh' is handled separately in scripts/04_init_gs_from_first_frame.py
    (it doesn't follow the GSBackend interface — it reads URDF + meshes
    directly, no rgb/cameras needed).
    """
    backend_name = gs_cfg.get("backend", "gsplat").lower()

    if backend_name == "mvsplat":
        ckpt = gs_cfg.get("mvsplat_checkpoint")
        if not ckpt:
            raise ValueError("backend=mvsplat requires `gs.mvsplat_checkpoint` in config")
        return MVSplatBackend(ckpt, device=device)

    if backend_name == "gsplat":
        return GsplatBackend(
            iters=gs_cfg.get("fallback_recon_iters", 3000),
            init_n_points=gs_cfg.get("fallback_init_n_points", 50_000),
            device=device,
        )

    raise ValueError(f"Unknown gs.backend: {backend_name!r}")


# ============================================================================
# .ply IO (compatible with standard 3DGS viewer / SIBR / SuperSplat)
# ============================================================================
def save_gs_to_ply(gs: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if gs is None:
        # Placeholder so downstream code finds a file
        with open(out_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
                    b"property float x\nproperty float y\nproperty float z\nend_header\n")
        return

    try:
        from plyfile import PlyData, PlyElement
    except ImportError:
        logger.error("plyfile not installed; saving placeholder")
        with open(out_path, "wb") as f:
            f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
                    b"property float x\nproperty float y\nproperty float z\nend_header\n")
        return

    means = gs["means"].astype(np.float32)
    scales = np.log(np.clip(gs["scales"], 1e-6, None)).astype(np.float32)
    quats = gs["quats"].astype(np.float32)
    opacities = _logit_np(gs["opacities"]).astype(np.float32).reshape(-1, 1)
    colors = gs["colors"].astype(np.float32)

    n = means.shape[0]
    dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
             ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
             ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
             ("opacity", "f4"),
             ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
             ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]
    arr = np.empty(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means.T
    arr["nx"] = arr["ny"] = arr["nz"] = 0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = colors.T
    arr["opacity"] = opacities[:, 0]
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales.T
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = quats.T

    el = PlyElement.describe(arr, "vertex")
    PlyData([el]).write(str(out_path))
    logger.info("Wrote %d gaussians to %s", n, out_path)


def _logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))
