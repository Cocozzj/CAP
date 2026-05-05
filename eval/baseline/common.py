"""Shared utilities for baselines: unified I/O format + per-trajectory metric
computation.  Every baseline writes its predictions in the same format so the
aggregator can compare them apples-to-apples in the main results table.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Output paths
# ══════════════════════════════════════════════════════════════════════

def baseline_output_dir(
    baselines_root: Path | str,
    baseline_name:  str,
    dataset:        str,                # "dataset_a" | "dataset_b"
    split:          str,                # "test_iid" | "test_ood_*" | ...
    traj_id:        str,
) -> Path:
    """Canonical per-trajectory output directory:

        runs/baselines/<baseline>/<dataset>/<split>/<traj_id>/
    """
    p = Path(baselines_root) / baseline_name / dataset / split / traj_id
    p.mkdir(parents=True, exist_ok=True)
    return p


# ══════════════════════════════════════════════════════════════════════
# 4DGS sequence I/O
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GS4DSequence:
    """A predicted 4DGS sequence — what every "scene-aware" baseline outputs.

    Pixel-only baselines (MAGVIT v2) produce no 4DGS — they skip this and only
    produce `pred_render.mp4` directly.

    Fields shape:
      mu      [T, N, 3]
      cov     [T, N, 3, 3]
      sh      [T, N, C_sh]    (default C_sh=48 = SH degree 3 × 3 channels)
      opacity [T, N, 1]
      scale   [T, N, 3]
    """
    mu:      np.ndarray
    cov:     np.ndarray
    sh:      np.ndarray
    opacity: np.ndarray
    scale:   np.ndarray

    @property
    def T(self) -> int:
        return int(self.mu.shape[0])

    @property
    def N(self) -> int:
        return int(self.mu.shape[1])

    def save(self, path: Path | str) -> None:
        path = Path(path)
        np.savez_compressed(
            path,
            mu=self.mu.astype(np.float32),
            cov=self.cov.astype(np.float32),
            sh=self.sh.astype(np.float32),
            opacity=self.opacity.astype(np.float32),
            scale=self.scale.astype(np.float32),
            T=self.T,
        )

    @classmethod
    def load(cls, path: Path | str) -> "GS4DSequence":
        z = np.load(path)
        return cls(
            mu=z["mu"], cov=z["cov"], sh=z["sh"],
            opacity=z["opacity"], scale=z["scale"],
        )


# ══════════════════════════════════════════════════════════════════════
# Per-trajectory metrics
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TrajMetrics:
    """Metrics for ONE predicted trajectory vs ground truth.

    Set unsupported metrics to None.  The aggregator handles None safely
    (treats as N/A in the final table).

    Categories (mapped to PDF §5.1):
      Trajectory       ade, fde, mpjpe                 (PDF #3, #4)
      Algebraic        closure_gap, inverse_gap         (PDF #1, #2; only Ours)
      Visual           psnr, ssim, lpips                (rendering quality)
      Physics          phys_wasserstein,                (PDF #11)
                       energy_violation,                (per-traj kinetic energy CV)
                       contact_violation,               (sub-metric of PhysConsis)
                       volume_violation                 (sub-metric of PhysConsis)
      Success          success                           (PDF #5; 0/1 per-traj)
      Diversity        action_diversity (D)              (PDF #9; Lev distance)
                       result_diversity (D_result)       (PDF #10; final state W)
    """
    # Trajectory
    ade:               Optional[float] = None
    fde:               Optional[float] = None
    mpjpe:             Optional[float] = None
    # Algebraic (only meaningful for methods that work in our codebook)
    closure_gap:       Optional[float] = None
    inverse_gap:       Optional[float] = None
    # Visual
    psnr:              Optional[float] = None
    ssim:              Optional[float] = None
    lpips:             Optional[float] = None
    # Physics (PDF metric #11 + sub-metrics)
    phys_wasserstein:  Optional[float] = None    # 1-W between pred and GT trajectories
    energy_violation:  Optional[float] = None    # KE coefficient of variation
    contact_violation: Optional[float] = None    # mean contact penetration depth
    volume_violation:  Optional[float] = None    # |1 - det(F)| volume preservation
    # Success
    success:           Optional[int]   = None
    # Diversity (PDF #9, #10) — populated by diversity_eval.py at the (split, baseline) level
    action_diversity:  Optional[float] = None    # mean Levenshtein D over text inputs
    result_diversity:  Optional[float] = None    # mean final-state Wasserstein
    # Free-form notes (e.g. "skipped — V=1 not supported")
    notes:             Optional[str]   = None

    def save(self, path: Path | str) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "TrajMetrics":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)


# ══════════════════════════════════════════════════════════════════════
# GT data accessors (per-trajectory)
# ══════════════════════════════════════════════════════════════════════

def load_meta_json(traj_dir: Path | str) -> Dict[str, Any]:
    """Load meta.json (contains task_name, obj_category, n_frames, etc.)."""
    with open(Path(traj_dir) / "meta.json") as f:
        return json.load(f)


def load_physics_json(traj_dir: Path | str) -> Optional[Dict[str, Any]]:
    """Load physics.json for Dataset-A (articulation info from SAPIEN simulator).

    Expected keys (subject to your dataset's exact format — adjust as needed):
      joint_type:  "revolute" | "prismatic"
      joint_axis:  [3] unit vector in world frame
      joint_origin: [3] point on axis
      joint_range: [min, max]   (radians for revolute, metres for prismatic)
      part_link:   index of the moving part (which Gaussians belong to it)

    Returns None if the file does not exist (Dataset-B has no GT physics).
    """
    p = Path(traj_dir) / "physics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_trajectory_npz(traj_dir: Path | str) -> Optional[Dict[str, np.ndarray]]:
    """Load trajectory.npz for Dataset-A (GT joint motion + object pose timeseries).

    Expected keys:
      joint_angles:  [T] float — joint position over time
      object_pose:   [T, 4, 4] — SE(3) world pose per frame (optional)
      part_mask:     [N] bool  — which Gaussian belongs to the moving part (optional)

    Returns None if the file does not exist (Dataset-B has no GT trajectory).
    """
    p = Path(traj_dir) / "trajectory.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ══════════════════════════════════════════════════════════════════════
# Iteration over a dataset split (manifest-driven)
# ══════════════════════════════════════════════════════════════════════

def iter_split_entries(
    manifest_path: Path | str,
    data_dir:      Path | str,
    split:         str,
):
    """Yield ``(traj_id, traj_dir, entry)`` for every trajectory in ``split``.

    `entry` is the raw JSON dict from manifest.json (keys: rel_dir, splits,
    task_name, obj_category, n_frames, ...).  This mirrors what DatasetA/B
    consume but lets baselines bypass the heavy multi-view-loading dataset
    class — most baselines only need init_gs.ply / meta.json / physics.json.
    """
    with open(manifest_path) as f:
        entries = json.load(f)["entries"]
    for e in entries:
        if split not in e.get("splits", []):
            continue
        rel = e["rel_dir"]
        traj_id  = Path(rel).name
        traj_dir = Path(data_dir) / rel
        yield traj_id, traj_dir, e
