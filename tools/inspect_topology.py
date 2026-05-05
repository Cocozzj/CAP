"""Inspect object-category distribution across a dataset for the Cross-Topology
experiment (train on rigid → test on soft / deformable objects).

The Cross-Topology column in the paper's main table tests whether our model,
trained only on RIGID articulated objects (drawers, cabinets, doors, etc.),
generalizes to SOFT / DEFORMABLE objects (Cloth, SoftToy) at test time.

This is an OOD-by-object-type evaluation that complements the per-split OOD
columns (unseen_pair, unseen_object).  Soft objects exercise the deformation
backend (PBD) while rigid objects exercise the articulated-joint backend.

Usage:

    python -m tools.inspect_topology \\
        --manifest dataset/dataset_a/manifest.json \\
        --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


# ──────────────────────────────────────────────────────────────────────
# Default category classification
# ──────────────────────────────────────────────────────────────────────
# These can be overridden via --soft-cats / --rigid-cats CLI flags.

DEFAULT_SOFT_CATS = {
    "Cloth",
    "SoftToy",
}

# Everything else is treated as rigid.  We do NOT enumerate rigid categories
# here so unseen rigid categories are still treated as rigid (default
# fallback).


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val",
                                                     "test_iid",
                                                     "test_ood_unseen_pair",
                                                     "test_ood_unseen_object",
                                                     "test_compositional_long"])
    p.add_argument("--soft-cats", nargs="+", default=sorted(DEFAULT_SOFT_CATS),
                   help="object categories to label as 'soft' (deformable)")
    args = p.parse_args(argv)

    with open(args.manifest) as f:
        entries = json.load(f)["entries"]

    soft_set = set(args.soft_cats)
    print(f"\n=== Soft categories: {sorted(soft_set)} ===\n")

    # Overall by category
    print("=== Object-category distribution (full manifest) ===")
    cats = Counter(e.get("obj_category", "_unknown") for e in entries)
    for c, n in cats.most_common():
        marker = "  [SOFT]" if c in soft_set else ""
        print(f"  {c:30s}  {n:5d}{marker}")

    # Per-split breakdown
    print("\n=== Per-split soft / rigid counts ===\n")
    print(f"{'Split':<35} {'Total':>8} {'Soft':>8} {'Rigid':>8} {'%Soft':>8}")
    print("-" * 70)
    for sp in args.splits:
        sub = [e for e in entries if sp in e.get("splits", [])]
        n_total = len(sub)
        n_soft  = sum(1 for e in sub if e.get("obj_category") in soft_set)
        n_rigid = n_total - n_soft
        pct = (100 * n_soft / n_total) if n_total else 0
        print(f"{sp:<35} {n_total:>8} {n_soft:>8} {n_rigid:>8} {pct:>7.1f}%")

    # Recommend cross-topology splits
    train_rigid = [e for e in entries
                    if "train" in e.get("splits", [])
                    and e.get("obj_category") not in soft_set]
    test_soft = [e for e in entries
                 if any(s in e.get("splits", []) for s in
                         ("test_iid", "test_ood_unseen_pair",
                          "test_ood_unseen_object", "val"))
                 and e.get("obj_category") in soft_set]

    print(f"\n=== Recommended Cross-Topology splits ===")
    print(f"  train_rigid (train  ∩ ¬soft):                   {len(train_rigid)} trajectories")
    print(f"  test_soft   (test_*  ∩  soft):                  {len(test_soft)} trajectories")

    if len(test_soft) < 30:
        print(f"\n  ⚠ Only {len(test_soft)} soft test trajectories — "
              f"high statistical noise in Cross-Topology column.")
        print(f"    Consider adding 'val' soft samples to test set, or "
              f"reporting median instead of mean.")
    if len(train_rigid) < 500:
        print(f"\n  ⚠ Only {len(train_rigid)} rigid training trajectories — "
              f"may underperform full-data baseline.")

    # Soft sample task breakdown
    soft_in_test = [e for e in entries
                     if any(s in e.get("splits", []) for s in
                             ("test_iid", "test_ood_unseen_pair",
                              "test_ood_unseen_object"))
                     and e.get("obj_category") in soft_set]
    if soft_in_test:
        print(f"\n=== Soft test samples breakdown ===")
        task_dist = Counter(e.get("task_name") for e in soft_in_test)
        for t, n in task_dist.most_common():
            print(f"  {t:30s}  {n}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
