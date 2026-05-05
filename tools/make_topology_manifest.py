"""Generate a Cross-Topology manifest:

  - tags ``train_rigid``  on entries already in ``train`` whose obj_category ∉ soft
  - tags ``test_soft``    on entries already in any test_* split whose obj_category ∈ soft

This lets you train Ours with:

    --manifest dataset/dataset_a/manifest_topology.json
    --split    train_rigid

and evaluate baselines on:

    --splits   test_soft

Usage:

    python -m tools.make_topology_manifest \\
        --manifest dataset/dataset_a/manifest.json \\
        --output   dataset/dataset_a/manifest_topology.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from tools.inspect_topology import DEFAULT_SOFT_CATS


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--output",   required=True,
                   help="output manifest path (extends entries with topology labels)")
    p.add_argument("--soft-cats", nargs="+", default=sorted(DEFAULT_SOFT_CATS),
                   help="object categories to label as 'soft' (deformable)")
    p.add_argument("--src-train-split", default="train")
    p.add_argument("--src-test-splits", nargs="+",
                   default=["test_iid", "test_ood_unseen_pair",
                            "test_ood_unseen_object"],
                   help="which existing test splits to draw soft test samples from")
    p.add_argument("--include-val", action="store_true",
                   help="also include val split's soft samples in test_soft "
                        "(use if too few soft trajectories in test splits)")
    args = p.parse_args(argv)

    with open(args.manifest) as f:
        raw = json.load(f)
    entries = raw["entries"]
    soft_set = set(args.soft_cats)

    test_src_splits = list(args.src_test_splits)
    if args.include_val:
        test_src_splits.append("val")

    n_train_rigid = 0
    n_test_soft   = 0
    n_total       = len(entries)

    for e in entries:
        cat = e.get("obj_category", "")
        is_soft = cat in soft_set
        cur_splits: List[str] = list(e.get("splits", []))

        if args.src_train_split in cur_splits and not is_soft:
            cur_splits.append("train_rigid")
            n_train_rigid += 1
        if any(s in cur_splits for s in test_src_splits) and is_soft:
            cur_splits.append("test_soft")
            n_test_soft += 1

        e["splits"] = sorted(set(cur_splits))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"\n=== Cross-Topology manifest written ===")
    print(f"  Output:                  {out_path}")
    print(f"  Soft categories:         {sorted(soft_set)}")
    print(f"  Total entries scanned:   {n_total}")
    print(f"\n  → train_rigid (train  ∩ ¬soft):  {n_train_rigid}")
    print(f"  → test_soft   (test_*  ∩  soft):  {n_test_soft}"
          f"  ({'+val included' if args.include_val else 'val NOT included'})")

    if n_test_soft < 30:
        print(f"\n  ⚠ Only {n_test_soft} soft test trajectories.")
        print(f"    Add --include-val to widen the test pool.")

    print(f"\nNext steps:")
    print(f"  1. Train Ours on rigid only:")
    print(f"     --manifest {out_path}")
    print(f"     --split    train_rigid")
    print(f"  2. Run baselines on test_soft:")
    print(f"     --manifest {out_path}")
    print(f"     --splits   test_soft")
    return 0


if __name__ == "__main__":
    sys.exit(main())
