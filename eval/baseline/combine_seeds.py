"""Combine per-seed Ours runs into a single ``ours`` summary with mean ± std.

After running ``ours/runner.py`` 3 times with ``--baseline-name`` set to
``ours_s0``, ``ours_s1``, ``ours_s2``, the directory layout is:

    runs/baselines/
        ours_s0/dataset_a/test_iid/<traj>/{pred_4dgs.npz, metrics.json}
        ours_s1/dataset_a/test_iid/<traj>/{pred_4dgs.npz, metrics.json}
        ours_s2/dataset_a/test_iid/<traj>/{pred_4dgs.npz, metrics.json}

After ``aggregate.py`` has filled metrics.json, this script reads the three
seed branches, computes per-(dataset,split) mean ± std across seeds, and
writes them into a synthesized ``ours/`` directory tree:

    runs/baselines/ours/dataset_a/test_iid/summary.json   # mean+std fields

``format_latex.py`` then reads this synthesized ``ours`` exactly like any
other baseline; the +std fields appear in the table cells as
``mean ± std``.

Usage:

    python -m eval.baseline.combine_seeds \\
        --output-root runs/baselines \\
        --seeds ours_s0 ours_s1 ours_s2 \\
        --combined-name ours
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


# Metric fields we average across seeds (must match TrajMetrics fields)
_FIELDS = (
    "ade", "fde", "mpjpe",
    "closure_gap", "inverse_gap",
    "psnr", "ssim", "lpips",
    "phys_wasserstein",
    "energy_violation", "contact_violation", "volume_violation",
    "success",
    "action_diversity", "result_diversity",
)


def _mean_std(values: List[float]) -> Dict[str, float]:
    """Return {mean, std, n}.  Ignores None entries."""
    xs = [float(v) for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return {"mean": None, "std": None, "n": 0}
    if n == 1:
        return {"mean": xs[0], "std": 0.0, "n": 1}
    mean = sum(xs) / n
    var  = sum((x - mean) ** 2 for x in xs) / (n - 1)   # sample variance
    return {"mean": mean, "std": var ** 0.5, "n": n}


def _read_per_traj_metrics(seed_dir: Path) -> Dict[str, Dict[str, float]]:
    """For one seed branch, return {traj_id: {field: value}}."""
    out: Dict[str, Dict[str, float]] = {}
    for mp in seed_dir.glob("*/metrics.json"):
        traj = mp.parent.name
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        out[traj] = {f: m.get(f) for f in _FIELDS if m.get(f) is not None}
    return out


def combine_split(
    seed_split_dirs: List[Path],
    out_split_dir:   Path,
) -> Dict:
    """Combine N seed branches for one (dataset, split) → write summary.json."""
    per_seed = [_read_per_traj_metrics(d) for d in seed_split_dirs]

    # All traj_ids that appear in at least one seed
    all_trajs = set()
    for d in per_seed:
        all_trajs |= set(d.keys())

    # Per-field, per-seed → mean within each seed first, then across seeds
    summary: Dict[str, Dict[str, float]] = {}
    n_seeds = len(per_seed)

    # Step 1: per-seed mean (across trajectories)
    seed_means: Dict[str, List[float]] = {f: [] for f in _FIELDS}
    for d in per_seed:
        for f in _FIELDS:
            vs = [m.get(f) for m in d.values() if m.get(f) is not None]
            if vs:
                seed_means[f].append(sum(vs) / len(vs))
            else:
                seed_means[f].append(None)

    # Step 2: across-seed mean ± std
    for f in _FIELDS:
        # Drop seeds where this metric was empty
        valid = [v for v in seed_means[f] if v is not None]
        summary[f] = _mean_std(valid)

    # Trajectory counts
    summary["_meta"] = {
        "n_seeds":          n_seeds,
        "n_trajs_per_seed": [len(d) for d in per_seed],
        "n_trajs_union":    len(all_trajs),
        "seed_dirs":        [str(d) for d in seed_split_dirs],
    }

    out_split_dir.mkdir(parents=True, exist_ok=True)
    (out_split_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="runs/baselines",
                   help="parent containing ``ours_s*`` subdirs and "
                        "(after this script) ``ours``")
    p.add_argument("--seeds", nargs="+",
                   default=["ours_s0", "ours_s1", "ours_s2"],
                   help="seed directory names under output-root")
    p.add_argument("--combined-name", default="ours",
                   help="output baseline name (becomes runs/baselines/<name>/)")
    p.add_argument("--datasets", nargs="+", default=["dataset_a", "dataset_b"])
    args = p.parse_args(argv)

    root = Path(args.output_root)
    seed_dirs = [root / s for s in args.seeds]
    missing = [d for d in seed_dirs if not d.exists()]
    if missing:
        print(f"✗ missing seed dirs: {missing}", file=sys.stderr)
        return 1

    out_root = root / args.combined_name
    print(f"⏬ combining {len(args.seeds)} seeds → {out_root}/")

    n_splits = 0
    for ds in args.datasets:
        seed_ds_dirs = [d / ds for d in seed_dirs]
        # Find all splits that exist in any seed
        all_splits = set()
        for d in seed_ds_dirs:
            if d.exists():
                all_splits |= {p.name for p in d.iterdir() if p.is_dir()}
        for split in sorted(all_splits):
            seed_split_dirs = [d / split for d in seed_ds_dirs if (d / split).exists()]
            if not seed_split_dirs:
                continue
            out_split = out_root / ds / split
            summary = combine_split(seed_split_dirs, out_split)
            n_splits += 1
            ade  = summary.get("ade", {}).get("mean")
            ade_std = summary.get("ade", {}).get("std")
            print(f"  ✓ {ds}/{split:30s}  "
                  f"ade={ade:.4f}±{ade_std:.4f}  "
                  f"(n_seeds={summary['_meta']['n_seeds']})"
                  if ade is not None else
                  f"  ✓ {ds}/{split:30s}  (no ADE)")

    print(f"\n=== combine_seeds complete: {n_splits} splits written under {out_root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
