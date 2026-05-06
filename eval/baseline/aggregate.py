"""Aggregator for the main results table.

Iterates over every (baseline, dataset, split, traj_id) tuple, loads each
trajectory's prediction, computes metrics vs GT, and assembles the
mean ± std summary table that drops directly into the paper's Table 1.

For metrics this script CAN compute (geometric, no rendering):
  ade, fde, mpjpe, success, energy_violation

For metrics this script CANNOT compute here (deferred):
  psnr, lpips, ssim          → render_metrics.py (uses gsplat)
  closure_gap, inverse_gap   → algebraic_metrics.py (uses our Executor)
  diversity                  → diversity.py (uses 10-sample inference)
  multi_view_consistency     → render_metrics.py (uses multi-view rendering)

These deferred metrics are loaded from per-baseline metrics.json files if
the corresponding scripts have already filled them in (they share the same
per-trajectory metrics.json file as a "scratchpad").

Usage:

    # Step 1 (required): compute geometric metrics from pred_4dgs.npz
    python -m eval.baseline.aggregate \\
        --baselines tamp_pddl physgaussian physdreamer magvit_v2 motiongpt ours \\
        --output-root runs/baselines \\
        --data-root  dataset \\
        --output     runs/main_table.json

    # Step 2 (optional): re-aggregate after rendering / algebraic eval has
    # populated psnr/lpips/closure_gap into metrics.json:
    python -m eval.baseline.aggregate --reaggregate \\
        --output-root runs/baselines \\
        --output     runs/main_table.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence

import numpy as np

from .common import GS4DSequence, TrajMetrics
from .metrics import compute_all_metrics, load_gt_centers


# ══════════════════════════════════════════════════════════════════════
# Per-trajectory metric computation
# ══════════════════════════════════════════════════════════════════════

def _load_init_mu(traj_dir: Path) -> Optional[np.ndarray]:
    """Read init_gs.ply mu only (no need for full SH/cov for geometric metrics).

    Falls back to None if plyfile isn't installed.
    """
    try:
        from plyfile import PlyData
    except ImportError:
        return None
    p = traj_dir / "init_gs.ply"
    if not p.exists():
        return None
    v = PlyData.read(str(p))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)


def _evaluate_trajectory(
    pred_npz_path:  Path,
    traj_dir:       Path,
    overwrite:      bool = False,
) -> Optional[TrajMetrics]:
    """Evaluate one trajectory.  Writes metrics.json next to pred_4dgs.npz.

    If metrics.json already has ade/fde filled and overwrite=False, skip.
    """
    out_dir = pred_npz_path.parent
    metrics_path = out_dir / "metrics.json"

    # Lazy skip: if geometric metrics already computed, leave the file alone
    if metrics_path.exists() and not overwrite:
        try:
            existing = TrajMetrics.load(metrics_path)
            if existing.ade is not None and existing.notes != "pending_eval":
                return existing
        except Exception:
            pass

    if not pred_npz_path.exists():
        return None

    seq = GS4DSequence.load(pred_npz_path)
    init_mu = _load_init_mu(traj_dir)

    gt_centers: Optional[np.ndarray] = None
    if init_mu is not None:
        # Sub-sample GT in `metrics.load_gt_centers` to match pred T
        gt_centers = load_gt_centers(traj_dir, init_mu, T_pred=seq.T)

    # Pred mu may have N != GT N (e.g. PhysGaussian sub-samples).  Truncate to
    # the smaller — `compute_all_metrics` handles this internally too.
    if gt_centers is not None and gt_centers.shape[1] != seq.mu.shape[1]:
        N_min = min(gt_centers.shape[1], seq.mu.shape[1])
        seq = GS4DSequence(
            mu=seq.mu[:, :N_min],
            cov=seq.cov[:, :N_min],
            sh=seq.sh[:, :N_min],
            opacity=seq.opacity[:, :N_min],
            scale=seq.scale[:, :N_min],
        )
        gt_centers = gt_centers[:, :N_min]

    m = compute_all_metrics(seq, gt_centers)

    # Preserve fields written by other eval scripts (psnr/lpips/closure_gap, etc.)
    if metrics_path.exists():
        try:
            prev = TrajMetrics.load(metrics_path)
            for fname in ("psnr", "ssim", "lpips", "closure_gap", "inverse_gap"):
                pv = getattr(prev, fname, None)
                if pv is not None and getattr(m, fname, None) is None:
                    setattr(m, fname, pv)
        except Exception:
            pass

    m.save(metrics_path)
    return m


# ══════════════════════════════════════════════════════════════════════
# Per-(baseline, dataset, split) aggregation
# ══════════════════════════════════════════════════════════════════════

_METRIC_FIELDS = (
    # Trajectory
    "ade", "fde", "mpjpe",
    # Visual (rendering quality)
    "psnr", "ssim", "lpips",
    # Algebraic (PDF #1, #2; only Ours)
    "closure_gap", "inverse_gap",
    # Physics (PDF #11 + sub-metrics)
    "phys_wasserstein", "energy_violation",
    "contact_violation", "volume_violation",
    # Success (PDF #5)
    "success",
    # Diversity (PDF #9, #10) — populated separately by diversity_eval.py
    "action_diversity", "result_diversity",
)


def _aggregate(values: List[float]) -> Dict[str, float]:
    """mean / std / n / median for a list of floats (drops Nones beforehand)."""
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0,
                "median": float("nan")}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "std": 0.0, "n": 1, "median": v}
    return {
        "mean":   float(mean(values)),
        "std":    float(stdev(values)),
        "n":      int(len(values)),
        "median": float(np.median(values)),
    }


def aggregate_one(
    baseline:    str,
    dataset:     str,
    split:       str,
    traj_dirs:   Sequence[Path],
    data_root:   Path,
    overwrite:   bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run per-trajectory eval, then collect per-metric statistics."""
    per_field: Dict[str, List[float]] = {f: [] for f in _METRIC_FIELDS}
    n_processed = 0
    n_skipped   = 0

    for traj_out_dir in traj_dirs:
        traj_id = traj_out_dir.name
        # Find the corresponding GT data dir
        # convention: data_root / dataset / "data" / traj_id
        gt_dir = data_root / dataset / "data" / traj_id
        if not gt_dir.exists():
            n_skipped += 1
            continue

        pred_path = traj_out_dir / "pred_4dgs.npz"
        if pred_path.exists():
            m = _evaluate_trajectory(pred_path, gt_dir, overwrite=overwrite)
        else:
            # No pred_4dgs.npz (e.g. MAGVIT v2 only writes pred_render.mp4).
            # Just read whatever metrics.json has.
            mp = traj_out_dir / "metrics.json"
            m = TrajMetrics.load(mp) if mp.exists() else None

        if m is None:
            n_skipped += 1
            continue

        n_processed += 1
        for f in _METRIC_FIELDS:
            v = getattr(m, f, None)
            if v is not None:
                per_field[f].append(float(v))

    summary = {f: _aggregate(per_field[f]) for f in _METRIC_FIELDS}

    # ── Pull split-level diversity (PDF #9) and physics-W (PDF #11) ──
    # diversity_eval.py writes diversity.json at <baseline>/<dataset>/<split>/
    # physics_wasserstein.py  writes physics_wasserstein.json at the same level
    split_dir = (traj_dirs[0].parent if traj_dirs else None)
    if split_dir is not None:
        diversity_json = split_dir / "diversity.json"
        if diversity_json.exists():
            try:
                with open(diversity_json) as f:
                    dj = json.load(f)
                # Promote D_mean to action_diversity for the table
                summary["action_diversity"] = {
                    "mean":   float(dj.get("D_mean", float("nan"))),
                    "std":    float(dj.get("D_std", 0.0)),
                    "n":      int(dj.get("n_trajs", 0)),
                    "median": float(dj.get("D_mean", float("nan"))),
                }
            except Exception:
                pass

        physw_json = split_dir / "physics_wasserstein.json"
        if physw_json.exists():
            try:
                with open(physw_json) as f:
                    pj = json.load(f)
                summary["phys_wasserstein"] = {
                    "mean":   float(pj.get("W_mean", float("nan"))),
                    "std":    float(pj.get("W_std", 0.0)),
                    "n":      int(pj.get("n_trajs", 0)),
                    "median": float(pj.get("W_mean", float("nan"))),
                }
            except Exception:
                pass

    summary["_meta"] = {
        "baseline":    baseline,
        "dataset":     dataset,
        "split":       split,
        "n_total":     int(len(traj_dirs)),
        "n_processed": int(n_processed),
        "n_skipped":   int(n_skipped),
    }
    return summary


# ══════════════════════════════════════════════════════════════════════
# Pretty-print main table
# ══════════════════════════════════════════════════════════════════════

def _fmt(s: Dict[str, float], fmt: str = ".4f") -> str:
    if s["n"] == 0 or np.isnan(s["mean"]):
        return "  N/A "
    return f"{s['mean']:{fmt}}±{s['std']:{fmt}}"


def print_table(table: Dict, output_path: Path | str | None = None) -> str:
    """Render the aggregated dict as a paper-friendly text table."""
    lines = []
    columns = ["baseline", "dataset", "split",
               "ADE↓", "FDE↓", "MPJPE↓",
               "PSNR↑", "LPIPS↓",
               "Clos↓", "Inv↓",
               "Energy↓", "Success"]
    header = " | ".join(f"{c:>14s}" for c in columns)
    lines.append(header)
    lines.append("-" * len(header))

    for key, summary in table.items():
        if key.startswith("_"):
            continue
        meta = summary["_meta"]
        row  = [
            meta["baseline"], meta["dataset"], meta["split"],
            _fmt(summary["ade"]),
            _fmt(summary["fde"]),
            _fmt(summary["mpjpe"]),
            _fmt(summary["psnr"], ".2f"),
            _fmt(summary["lpips"]),
            _fmt(summary["closure_gap"]),
            _fmt(summary["inverse_gap"]),
            _fmt(summary["energy_violation"]),
            _fmt(summary["success"], ".3f"),
        ]
        lines.append(" | ".join(f"{v:>14s}" for v in row))

    out = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(out + "\n")
    return out


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+",
                   default=["tamp_pddl", "physgaussian", "svd",
                            "magvit_v2", "motiongpt", "ours"],
                   help="which baselines to aggregate (paper's 5 external "
                        "baselines + Ours).  'svd' is a generic video-"
                        "diffusion baseline — see eval/baseline/svd/README.md "
                        "(PhysDreamer is excluded due to its per-scene "
                        "optimization cost being infeasible at our scale).")
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--data-root", default="dataset",
                   help="root containing dataset_a/data/<traj>, dataset_b/data/<traj>")
    p.add_argument("--datasets", nargs="+", default=["dataset_a", "dataset_b"])
    p.add_argument("--splits",   nargs="+", default=None,
                   help="if None, auto-detect from output dirs")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute geometric metrics even if metrics.json exists")
    p.add_argument("--output", default="runs/main_table.json")
    p.add_argument("--text-output", default="runs/main_table.txt")
    args = p.parse_args(argv)

    out_root  = Path(args.output_root)
    data_root = Path(args.data_root)

    table: Dict = {}
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
                traj_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
                if not traj_dirs:
                    continue
                key = f"{baseline}::{dataset}::{split}"
                table[key] = aggregate_one(
                    baseline, dataset, split, traj_dirs,
                    data_root=data_root, overwrite=args.overwrite,
                )
                meta = table[key]["_meta"]
                print(f"[{baseline:14s}] {dataset}/{split:30s}  "
                      f"n={meta['n_processed']}/{meta['n_total']}  "
                      f"skipped={meta['n_skipped']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(table, f, indent=2, default=str)

    text = print_table(table, args.text_output)
    print()
    print(text)
    print(f"\n✓ wrote {args.output}")
    print(f"✓ wrote {args.text_output}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
