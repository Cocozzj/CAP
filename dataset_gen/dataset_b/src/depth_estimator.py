"""DepthAnything v2 wrapper for monocular depth estimation.

We use the HuggingFace `transformers` distribution. The model outputs an
inverse-depth (disparity-like) map; we convert to a coarse metric depth via
median normalization to a typical hand-object workspace scale (~1.5 m).

This metric scale is a HEURISTIC. Real metric depth would require a single
calibration target per video; for our purposes (init_gs back-projection
only), the relative scale is what matters — pixels close to the camera get
small depth, pixels far get large depth, and the resulting point cloud has
a sensible spatial distribution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


_MODEL_VARIANTS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


class DepthAnythingV2:
    """Lazy-load DepthAnything v2 via HuggingFace transformers.

    Args:
        variant: 'vits' (~25M, fast), 'vitb' (~98M), 'vitl' (~336M, best).
        device:  'cuda' / 'cuda:0' / 'cpu'
        scene_scale_m: median depth target in meters (default 1.5).
    """

    def __init__(
        self,
        variant: str = "vitl",
        device: str = "cuda",
        scene_scale_m: float = 1.5,
    ):
        self.variant = variant
        self.device = device
        self.scene_scale_m = scene_scale_m
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(
                "transformers is required. Install: "
                "pip install 'transformers>=4.40' accelerate"
            ) from e
        if self.variant not in _MODEL_VARIANTS:
            raise ValueError(f"Unknown variant {self.variant!r}; expected one of {list(_MODEL_VARIANTS)}")
        model_id = _MODEL_VARIANTS[self.variant]
        logger.info("Loading DepthAnything-v2 (%s) on %s", model_id, self.device)
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()

    @staticmethod
    def _to_metric(disparity: np.ndarray, scene_scale_m: float) -> np.ndarray:
        """Convert raw disparity-like output to a coarse metric depth in meters.

        Steps:
          1. clip very small disparities to avoid 1/0
          2. metric_depth = 1 / disparity  (relative)
          3. divide by median, multiply by scene_scale_m
        """
        d = np.asarray(disparity, dtype=np.float32)
        d = np.clip(d, a_min=1e-3, a_max=None)
        depth = 1.0 / d
        med = float(np.median(depth))
        if med <= 1e-6:
            return depth
        return depth * (scene_scale_m / med)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Predict metric depth (H, W) float32 meters from a single RGB frame.

        rgb: (H, W, 3) uint8 in [0, 255]
        """
        self._load()
        import torch

        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

        H, W = rgb.shape[:2]
        inputs = self._processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model(**inputs)
            disp = out.predicted_depth  # (1, h, w) tensor, possibly resized
            # Resize back to original resolution
            disp = torch.nn.functional.interpolate(
                disp.unsqueeze(1), size=(H, W), mode="bicubic", align_corners=False,
            ).squeeze(1).squeeze(0)
            disp = disp.detach().cpu().numpy()
        return self._to_metric(disp, self.scene_scale_m)
