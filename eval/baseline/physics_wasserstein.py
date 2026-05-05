"""Trajectory Physics Deviation (PDF metric #11): Wasserstein / DTW
distance between predicted Gaussian-center trajectory and the GT physics
simulation trajectory in state space.

PDF excerpt:
    "选取涉及物理规律的任务，例如仅训练推方块的情况下测试推重物。
     比较模型生成的物体运动轨迹和真实物理模拟的轨迹差距，
     用 Wasserstein 距离 或 DTW (dynamic time warping) 在状态空间计算。
     越小说明即使没见过也合乎物理。"

Implementation:
    For each predicted 4DGS trajectory we already have mu_t [T, N, 3].
    The GT trajectory comes from object_pose_world in trajectory.npz applied
    to init_gs (this is exactly what eval/baseline/metrics.py constructs).
    We compute:
        W₁(pred_centroid_traj, gt_centroid_traj)
    where centroid_traj is [T, 3] (mean Gaussian position per frame).

    This is the 1-Wasserstein distance between two T-step paths in ℝ³,
    treating each timestep as an empirical measure of mass (T uniform points
    on the path).  Use POT library if available, else fall back to a sorted
    1-D approximation per axis.

Output: writes ``physics_wasserstein.json`` per (baseline, dataset, split)
with mean / std of W across trajectories.

Usage:

    python -m eval.baseline.physics_wasserstein \\
        --baselines tamp_rule physgaussian flat_vqvae _4dgs ours \\
        --output-root runs/baselines \\
        --data-root  dataset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import List, Optional

import numpy as np

from .common import GS4DSequence


def _try_import_pot():
    try:
        import ot
        return ot
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════
# Distance metrics
# ══════════════════════════════════════════════════════════════════════

def wasserstein_1d_axis(pred_path: np.ndarray, gt_path: np.ndarray) -> float:
    """1-Wasserstein per axis, then average — exact closed form via sorting.

    Both paths are [T, 3].  W₁ between sorted x-coords + sorted y-coords +
    sorted z-coords / 3.  This is a standard cheap approximation when the
    paths have the same number of samples (T_pred == T_gt).
    """
    if pred_path.shape != gt_path.shape:
        T = min(pred_path.shape[0], gt_path.shape[0])
        pred_path = pred_path[:T]
        gt_path   = gt_path[:T]
    w_per_axis = []
    for ax in range(3):
        a = np.sort(pred_path[:, ax])
        b = np.sort(gt_path[:, ax])
        w_per_axis.append(float(np.abs(a - b).mean()))
    return float(np.mean(w_per_axis))


def wasserstein_full(pred_path: np.ndarray, gt_path: np.ndarray, ot=None) -> float:
    """Full 1-Wasserstein in 3D using POT's emd2 (more accurate than per-axis).

    pred_path, gt_path: [T, 3] — uniform empirical distributions.
    """
    if ot is None:
        return wasserstein_1d_axis(pred_path, gt_path)
    T_p = pred_path.shape[0]
    T_g = gt_path.shape[0]
    a = np.full(T_p, 1.0 / T_p, dtype=np.float64)
    b = np.full(T_g, 1.0 / T_g, dtype=np.float64)
    M = np.linalg.norm(pred_path[:, None, :] - gt_path[None, :, :], axis=-1).astype(np.float64)
    return float(ot.emd2(a, b, M))


def dtw_distance(pred_path: np.ndarray, gt_path: np.ndarray) -> float:
    """Dynamic Time Warping distance between two ℝ³ paths.

    Allows differing T_pred / T_gt by warping along time.  Returns the
    average per-step cost of the optimal warp.  Standard O(T²) DP.
    """
    T_p = int(pred_path.shape[0])
    T_g = int(gt_path.shape[0])
    INF = float("inf")
    cost = np.full((T_p + 1, T_g + 1), INF, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, T_p + 1):
        for j in range(1, T_g + 1):
            d = float(np.linalg.norm(pred_path[i - 1] - gt_path[j - 1]))
            cost[i, j] = d + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    # Normalize by warp path length (≈ max(T_p, T_g))
    return float(cost[T_p, T_g] / max(T_p + T_g, 1))


# ══════════════════════════════════════════════════════════════════════
# Centroid trajectory extraction
# ══════════════════════════════════════════════════════════════════════

def gs_centroid_trajectory(seq: GS4DSequence) -> np.ndarray:
    """[T, N, 3] Gaussian centers → [T, 3] per-frame centroid."""
    return seq.mu.mean(axis=1).astype(np.float32)


def gt_centroid_trajectory(traj_dir: Path, T: int = 30) -> Optional[np.ndarray]:
    """Read object_pose_world from trajectory.npz, return centroid path [T, 3].

    For our data, object_pose_world[:, :3] IS the moving object's centroid
    at each frame.  We sub-sample to T frames evenly.
    """
    p = traj_dir / "trajectory.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "object_pose_world" not in z.files:
        return None
    poses = z["object_pose_world"].astype(np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7 or poses.shape[0] < 2:
        return None
    idx = np.linspace(0, poses.shape[0] - 1, T, dtype=int)
    return poses[idx, :3]


# ══════════════════════════════════════════════════════════════════════
# Per-trajectory eval
# ══════════════════════════════════════════════════════════════════════

def evaluate_one(
    pred_npz_path: Path, traj_dir: Path,
    method: str = "wasserstein",
    T: int = 30,
    ot=None,
) -> Optional[float]:
    if not pred_npz_path.exists():
        return None
    seq = GS4DSequence.load(pred_npz_path)
    pred = gs_centroid_trajectory(seq)
    gt   = gt_centroid_trajectory(traj_dir, T=pred.shape[0])
    if gt is None:
        return None
    if method == "wasserstein":
        return wasserstein_full(pred, gt, ot=ot)
    if method == "wasserstein_1d":
        return wasserstein_1d_axis(pred, gt)
    if method == "dtw":
        return dtw_distance(pred, gt)
    raise ValueError(f"unknown method {method!r}")


# ══════════════════════════════════════════════════════════════════════
# Main aggregation
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+",
                   default=["tamp_pddl", "physgaussian", "physdreamer",
                            "motiongpt", "ours"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--data-root",   default="dataset")
    p.add_argument("--datasets",    nargs="+", default=["dataset_a"])
    p.add_argument("--splits",      nargs="+", default=None,
                   help="if None, auto-detect from output dirs")
    p.add_argument("--method", choices=["wasserstein", "wasserstein_1d", "dtw"],
                   default="wasserstein",
                   help="distance: wasserstein (POT, accurate), "
                        "wasserstein_1d (per-axis sorting, no deps), "
                        "dtw (warp-tolerant)")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    ot = _try_import_pot() if args.method == "wasserstein" else None
    if args.method == "wasserstein" and ot is None:
        print("⚠ POT (python-optimal-transport) not installed; "
              "falling back to wasserstein_1d (per-axis approximation).",
              file=sys.stderr)
        args.method = "wasserstein_1d"

    out_root  = Path(args.output_root)
    data_root = Path(args.data_root)
    n_total = 0
    t0 = time.time()

    for baseline in args.baselines:
        for dataset in args.datasets:
            base = out_root / baseline / dataset
            if not base.exists():
                continue
            splits = args.splits or [d.name for d in sorted(base.iterdir()) if d.is_dir()]
            for split in splits:
                split_dir = base / split
                if not split_dir.exists():
                    continue
                per_traj: List[float] = []
                for traj_out in sorted(split_dir.iterdir()):
                    if not traj_out.is_dir():
                        continue
                    if args.limit is not None and len(per_traj) >= args.limit:
                        break
                    n_total += 1
                    gt_dir = data_root / dataset / "data" / traj_out.name
                    if not gt_dir.exists():
                        continue
                    pred_npz = traj_out / "pred_4dgs.npz"
                    val = evaluate_one(pred_npz, gt_dir, method=args.method,
                                         T=args.T, ot=ot)
                    if val is not None and np.isfinite(val):
                        per_traj.append(val)

                if per_traj:
                    W_mean = float(mean(per_traj))
                    W_std  = float(stdev(per_traj)) if len(per_traj) > 1 else 0.0
                else:
                    W_mean = W_std = float("nan")

                split_dir.mkdir(parents=True, exist_ok=True)
                with open(split_dir / "physics_wasserstein.json", "w") as f:
                    json.dump({
                        "method":      args.method,
                        "W_mean":      W_mean,
                        "W_std":       W_std,
                        "n_trajs":     len(per_traj),
                        "per_traj_W":  per_traj,
                    }, f, indent=2)
                print(f"[{baseline:14s}] {dataset}/{split:30s}  "
                      f"W = {W_mean:.4f} ± {W_std:.4f}   "
                      f"(method={args.method}, n={len(per_traj)})")

    print(f"\nPhysics-Wasserstein done in {time.time()-t0:.1f}s, "
          f"{n_total} trajectories scanned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
