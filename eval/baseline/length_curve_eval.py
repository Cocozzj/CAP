"""Length-vs-Success curve (PDF metric #8): plot success rate vs sequence
length, report L_max where success drops below 50%.

Implementation:
    Compositional tasks in our manifest are named like:
        "open"
        "comp:close_open"            (2 atomic steps)
        "comp:close_open_close"      (3 steps)
        "comp:open_close_open_close" (4 steps)
        "comp:pull_push_pull_push"   (4 steps)
        ...

    Step count = 1 if not "comp:..."  else len(name.split(":")[1].split("_"))

    For each baseline, group its trajectories by step count, compute success
    rate per group, write a curve.

Usage:

    # Aggregator must already have populated metrics.json per trajectory
    python -m eval.baseline.length_curve_eval \\
        --baselines tamp_rule physgaussian flat_vqvae _4dgs ours \\
        --output-root runs/baselines \\
        --data-root  dataset \\
        --datasets   dataset_a \\
        --output     runs/length_curve.json

    # (optional) Render plot
    python -m eval.baseline.length_curve_eval \\
        --plot runs/length_curve.json \\
        --output-png runs/length_curve.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from .common import TrajMetrics


# ══════════════════════════════════════════════════════════════════════
# Step counting
# ══════════════════════════════════════════════════════════════════════

_COMP_PATTERN = re.compile(r"^comp:(.+)$")


def task_step_count(task_name: str) -> int:
    """Number of atomic steps in a task name.

    Single-verb tasks count as 1.  Composite tasks (``comp:close_open``)
    count by underscore-separated atomic verbs.
    """
    if task_name is None:
        return 1
    m = _COMP_PATTERN.match(task_name)
    if m is None:
        return 1
    parts = m.group(1).split("_")
    # Drop "more" suffix (e.g. "open_open_more" → 2-step ramp-up)
    parts = [p for p in parts if p and p != "more"]
    return max(len(parts), 1)


# ══════════════════════════════════════════════════════════════════════
# Per-(baseline, dataset, split) length curve
# ══════════════════════════════════════════════════════════════════════

def collect_length_groups(
    baseline_root: Path,
    data_root:     Path,
    dataset:       str,
    splits:        Optional[List[str]] = None,
) -> Dict[int, List[float]]:
    """Walk per-trajectory metrics.json files, group success values by step count.

    Returns {step_count: [success_value, ...]}
    """
    base = baseline_root / dataset
    if not base.exists():
        return {}
    splits = splits or [d.name for d in sorted(base.iterdir()) if d.is_dir()]

    groups: Dict[int, List[float]] = defaultdict(list)
    for sp in splits:
        split_dir = base / sp
        if not split_dir.exists():
            continue
        for traj_out in sorted(split_dir.iterdir()):
            if not traj_out.is_dir():
                continue
            traj_id = traj_out.name
            # Need GT meta.json for task_name → step count
            gt_meta = data_root / dataset / "data" / traj_id / "meta.json"
            if not gt_meta.exists():
                continue
            with open(gt_meta) as f:
                meta = json.load(f)
            steps = task_step_count(meta.get("task_name"))

            # Read metrics.json (filled by aggregate.py)
            mp = traj_out / "metrics.json"
            if not mp.exists():
                continue
            try:
                m = TrajMetrics.load(mp)
            except Exception:
                continue
            if m.success is None:
                continue
            groups[steps].append(float(m.success))

    return dict(groups)


def compute_curve(groups: Dict[int, List[float]]) -> List[Dict]:
    """Convert {step_count: [succ, ...]} → sorted list of {step, succ_mean, n}."""
    out = []
    for steps in sorted(groups.keys()):
        vals = groups[steps]
        if not vals:
            continue
        out.append({
            "step":      int(steps),
            "succ_mean": float(mean(vals)),
            "n":         int(len(vals)),
        })
    return out


def find_L_max(curve: List[Dict], threshold: float = 0.5) -> Optional[int]:
    """Largest step count where succ_mean still >= threshold (PDF: 50%)."""
    L = None
    for entry in curve:
        if entry["succ_mean"] >= threshold:
            L = entry["step"]
        else:
            break    # once below threshold, monotone-degrading assumption
    return L


# ══════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════

_BASELINE_STYLE = {
    "tamp_pddl":    {"label": "TAMP (PDDLStream)", "linestyle": ":",  "marker": "x"},
    "physgaussian": {"label": "PhysGaussian",       "linestyle": "-",  "marker": "s"},
    "physdreamer":  {"label": "PhysDreamer",        "linestyle": "--", "marker": "D"},
    "magvit_v2":    {"label": "MAGVIT-v2",          "linestyle": ":",  "marker": "^"},
    "motiongpt":    {"label": "MotionGPT",          "linestyle": "--", "marker": "o"},
    "ours":         {"label": "Ours",               "linestyle": "-",  "marker": "*"},
    # Legacy
    "flat_vqvae":   {"label": "Flat VQ-VAE",        "linestyle": "--", "marker": "o"},
    "tamp_rule":    {"label": "TAMP-rule",          "linestyle": ":",  "marker": "x"},
    "_4dgs":        {"label": "4D-GS",              "linestyle": ":",  "marker": "v"},
}


def render_plot(curves: Dict, out_png: Path, threshold: float = 0.5) -> None:
    """One line per baseline; x = step count, y = success rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("✗ matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for baseline, dat in curves.items():
        for dataset, by_dataset in dat.items():
            curve = by_dataset["curve"]
            if not curve:
                continue
            xs = [e["step"]      for e in curve]
            ys = [e["succ_mean"] for e in curve]
            style = _BASELINE_STYLE.get(baseline, {})
            label = style.get("label", baseline) + f" ({dataset})"
            ax.plot(xs, ys,
                     linestyle=style.get("linestyle", "-"),
                     marker=style.get("marker", "o"),
                     label=label,
                     linewidth=1.5)
    ax.axhline(threshold, color="grey", linestyle=":", alpha=0.6,
                label=f"{threshold*100:.0f}% threshold")
    ax.set_xlabel("Sequence length (atomic steps)")
    ax.set_ylabel("Success rate")
    ax.set_title("Success rate vs trajectory length")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"✓ wrote {out_png}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+",
                   default=["tamp_pddl", "physgaussian", "physdreamer",
                            "magvit_v2", "motiongpt", "ours"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--data-root",   default="dataset")
    p.add_argument("--datasets",    nargs="+", default=["dataset_a"])
    p.add_argument("--splits",      nargs="+", default=None)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="L_max threshold (default 0.5 per PDF)")
    p.add_argument("--output",     default="runs/length_curve.json")

    # Plot mode (re-uses an existing length_curve.json)
    p.add_argument("--plot",       default=None,
                   help="if set, read this length_curve.json and render PNG; "
                        "skips the data-collection phase")
    p.add_argument("--output-png", default="runs/length_curve.png")

    args = p.parse_args(argv)

    if args.plot:
        with open(args.plot) as f:
            curves = json.load(f)
        render_plot(curves, Path(args.output_png), threshold=args.threshold)
        return 0

    out_root  = Path(args.output_root)
    data_root = Path(args.data_root)
    curves: Dict[str, Dict[str, Dict]] = {}
    t0 = time.time()

    for baseline in args.baselines:
        baseline_root = out_root / baseline
        if not baseline_root.exists():
            continue
        curves[baseline] = {}
        for dataset in args.datasets:
            groups = collect_length_groups(baseline_root, data_root, dataset, args.splits)
            curve  = compute_curve(groups)
            L_max  = find_L_max(curve, threshold=args.threshold)
            curves[baseline][dataset] = {
                "curve": curve,
                "L_max": L_max,
            }
            n_groups = len(curve)
            n_trajs  = sum(e["n"] for e in curve)
            print(f"[{baseline:14s}] {dataset:12s}  L_max={L_max}  "
                  f"({n_groups} length groups, {n_trajs} trajs)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(curves, f, indent=2)

    # Always render plot at end as well
    render_plot(curves, Path(args.output_png), threshold=args.threshold)

    print(f"\n✓ wrote {args.output}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
