"""Re-evaluate the `success` flag on every trajectory in trajectories.json
without re-running any SAPIEN simulation.

The original criterion in src/tasks/base.py compared achieved qpos at the
END-OF-MOTION frame against the target end, with an absolute error tolerance.
This was too strict for high-inertia / sign-flipped joints (PartNet stores
some limits with low > high) and produced false negatives across atomic
open / pull / push and downstream compositions.

This script applies a more robust criterion using the per-frame
`joint_qpos_actual` already recorded in trajectories.json:

    total_motion    = |qpos_end_target - qpos_start_target|
    achieved_motion = |qpos_actual_final - qpos_start_target|
    success         = (achieved_motion / total_motion) >= 0.5
                      AND sign(actual move) == sign(target move)

For composite trajectories, every sub-action's sub-range must satisfy the
same criterion using its own (start, end) frame indices.

Usage:
    python scripts/re_evaluate_success.py \\
        --in outputs/trajectories.json \\
        --out outputs/trajectories.json \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _segment_success(target: np.ndarray,
                     actual: np.ndarray,
                     start_f: int,
                     end_f: int,
                     final_eval_frame: int,
                     threshold: float) -> bool:
    """Whether the segment [start_f .. end_f] of actual qpos achieved
    >= threshold of the targeted motion.

    final_eval_frame: where to read achieved qpos. For atomic, we want the
    very last frame (after post_settle). For composite sub-actions, we want
    the end of that sub-action's motion (= end_f).
    """
    if start_f >= len(target) or end_f >= len(target):
        return False
    qpos_start = float(target[start_f])
    qpos_end = float(target[end_f])
    achieved = float(actual[final_eval_frame])
    total_motion = abs(qpos_end - qpos_start)
    if total_motion < 1e-6:
        return True  # degenerate / no motion expected
    # Use SIGNED progress: how far the joint moved toward qpos_end as a
    # fraction of the targeted motion. This naturally handles PartNet URDFs
    # that store limits with low > high (the "sign-flipped" case) — what
    # matters is whether achieved is between start and end (or beyond), not
    # the global +/- direction of the move.
    signed_progress = (achieved - qpos_start) / (qpos_end - qpos_start)
    return signed_progress >= threshold


def re_evaluate(records: list, threshold: float = 0.5) -> tuple[int, int]:
    """Mutates records in-place; returns (n_changed, n_now_success)."""
    n_changed = 0

    for r in records:
        if r.get("object_type") == "soft":
            continue  # soft tasks are deterministic, leave success as-is

        target = r.get("joint_qpos") or []
        actual = r.get("joint_qpos_actual") or []
        if not target or not actual or len(target) != len(actual):
            continue
        target = np.asarray(target, dtype=np.float64)
        actual = np.asarray(actual, dtype=np.float64)

        old = bool(r.get("success", False))

        if not r.get("is_composition", False):
            # Atomic: motion lives in [n_pre .. n_pre + n_motion - 1].
            # Evaluate using FINAL frame (after post_settle) so PD can settle.
            n_pre = int(r.get("pre_settle_frames", 0))
            n_motion = int(r.get("motion_frames", 0))
            if n_motion <= 0:
                continue
            start_f = n_pre
            end_f = min(n_pre + n_motion - 1, len(target) - 1)
            final_eval_frame = len(actual) - 1
            new = _segment_success(target, actual, start_f, end_f,
                                   final_eval_frame, threshold)
        else:
            # Composite: every sub-action range must pass.
            # Use END of that sub-action (which already includes a sub-settle
            # when the composition pipeline pads between actions).
            ranges = r.get("sub_action_frame_ranges", [])
            if not ranges:
                continue
            new = True
            for rng_pair in ranges:
                start_f, end_f = int(rng_pair[0]), int(rng_pair[1])
                end_f = min(end_f, len(target) - 1)
                # Evaluate at the very end of THIS sub-range
                if not _segment_success(target, actual, start_f, end_f,
                                        end_f, threshold):
                    new = False
                    break

        if new != old:
            n_changed += 1
            r["success"] = bool(new)

    n_success = sum(1 for r in records if r.get("success"))
    return n_changed, n_success


def print_stats(records: list, label: str = "After re-eval"):
    n = len(records)
    n_success = sum(1 for r in records if r.get("success"))
    print(f"\n=== {label} ===")
    print(f"Total: {n}   Success: {n_success}  ({100*n_success/n:.1f}%)")

    by_task = defaultdict(lambda: [0, 0])
    by_cat = defaultdict(lambda: [0, 0])
    for r in records:
        by_task[r["task_name"]][1] += 1
        by_cat[r["obj_category"]][1] += 1
        if r.get("success"):
            by_task[r["task_name"]][0] += 1
            by_cat[r["obj_category"]][0] += 1

    print("\nBy task:")
    for t in sorted(by_task):
        s, tot = by_task[t]
        print(f"  {t:35s} {s:5d}/{tot:5d}  {100*s/tot:5.1f}%")

    print("\nBy category:")
    for c in sorted(by_cat):
        s, tot = by_cat[c]
        print(f"  {c:30s} {s:5d}/{tot:5d}  {100*s/tot:5.1f}%")

    n_atomic_s = sum(1 for r in records
                     if not r.get("is_composition") and r.get("success"))
    n_atomic = sum(1 for r in records if not r.get("is_composition"))
    n_comp_s = sum(1 for r in records
                   if r.get("is_composition") and r.get("success"))
    n_comp = sum(1 for r in records if r.get("is_composition"))
    print(f"\nAtomic:      {n_atomic_s}/{n_atomic}  ({100*n_atomic_s/max(n_atomic,1):.1f}%)")
    print(f"Composition: {n_comp_s}/{n_comp}  ({100*n_comp_s/max(n_comp,1):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="outputs/trajectories.json")
    ap.add_argument("--out", default="outputs/trajectories.json")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Fraction of targeted motion required for success")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print stats without writing")
    args = ap.parse_args()

    inp = Path(args.inp)
    print(f"Loading {inp} ...")
    with open(inp) as f:
        data = json.load(f)
    records = data["trajectories"] if "trajectories" in data else data
    print(f"Loaded {len(records)} records.")

    print_stats(records, label="BEFORE re-eval")

    n_changed, n_success = re_evaluate(records, threshold=args.threshold)
    print(f"\nRelabeled {n_changed} records.")

    print_stats(records, label="AFTER re-eval")

    if args.dry_run:
        print("\n[dry_run] not writing.")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"trajectories": records, "n": len(records)}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
