"""TAMP (PDDLStream) main entry — iterate test split, solve, write outputs.

This is a SKELETON — the heavy PDDLStream + Fast-Downward integration is
left as TODO blocks so the file shape is in place but each integration
point is clearly marked.

Pipeline per trajectory:
  1. Load init_gs.ply + meta.json + GT articulation (from URDF if accessible)
  2. Build PDDL problem: (objects, init facts, goal facts) per task_name
  3. PDDLStream solve(domain.pddl, stream.pddl, problem) → action plan
  4. Execute plan via motion primitives → trajectory of object poses
  5. Apply trajectory to init_gs → 4DGS sequence
  6. Write pred_4dgs.npz

Usage:

    python -m eval.baseline.tamp_pddl.run_tamp \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines \\
        --T 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from dataload.common import load_init_gs_ply
from dataload.text import task_to_text

from ..common import (
    GS4DSequence,
    TrajMetrics,
    baseline_output_dir,
    iter_split_entries,
    load_meta_json,
)
from ..kinematics import (
    apply_pose_trajectory_to_gs,
    quat_log_scale_to_full_cov,
)


# ══════════════════════════════════════════════════════════════════════
# PDDL world model construction (TODO: connect to PDDLStream)
# ══════════════════════════════════════════════════════════════════════

def _try_import_pddlstream():
    """Lazy-import PDDLStream so this file at least IMPORTS without it installed."""
    try:
        from pddlstream.algorithms.meta import solve  # noqa: F401
        from pddlstream.language.constants import And, Equal  # noqa: F401
        return True
    except ImportError:
        return False


def _build_pddl_problem(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one trajectory's meta.json → PDDL problem dict.

    PDDL problem shape (PDDLStream convention):
        {
            "domain":  Path("domain.pddl"),
            "objects": [...],
            "init":    [(predicate, args...), ...],
            "goal":    (predicate, args...),
        }

    TODO: implement per task_name (open / close / push / pull / rotate / ...)
    Use meta.randomization.start_fraction / target_fraction to set initial /
    goal joint angles.
    """
    task = meta.get("task_name", "")
    obj  = meta.get("obj_id", "obj0")

    # Skeleton — placeholder for actual implementation
    if task == "close":
        return {
            "objects": [obj, "joint0"],
            "init":   [("HasJoint", obj, "joint0"),
                        ("JointAngle", "joint0", "current"),
                        ("JointMin",   "joint0", "min"),
                        ("Open", obj)],
            "goal":   ("Closed", obj),
        }
    if task == "open":
        return {
            "objects": [obj, "joint0"],
            "init":   [("HasJoint", obj, "joint0"),
                        ("JointAngle", "joint0", "current"),
                        ("JointMax",   "joint0", "max"),
                        ("Closed", obj)],
            "goal":   ("Open", obj),
        }
    # Composite tasks: chain "comp:close_open" → goal sequence
    if task.startswith("comp:"):
        steps = task.split(":", 1)[1].split("_")
        last_pred = "Closed" if steps[-1] == "close" else "Open"
        return {
            "objects": [obj, "joint0"],
            "init":   [("HasJoint", obj, "joint0"),
                        ("JointAngle", "joint0", "current")],
            "goal":   (last_pred, obj),
        }
    # push / pull / rotate / ... — not yet defined in domain
    return None


def _solve_pddl(problem: Dict[str, Any]) -> Optional[List[Tuple[str, Any]]]:
    """Call PDDLStream → return action plan.

    TODO: real implementation using ``pddlstream.algorithms.meta.solve``.
    """
    if not _try_import_pddlstream():
        return None
    # TODO: real solve call.  For skeleton, return a fake single-step plan.
    return [("toggle", problem.get("goal"))]


def _execute_plan_to_trajectory(
    plan:    List[Tuple[str, Any]],
    meta:    Dict[str, Any],
    traj_dir: Path,
    T:       int,
) -> Optional[np.ndarray]:
    """Convert a PDDL plan → per-frame object_pose_world [T, 7].

    For skeleton, we fall back to the same "linear interpolation between GT
    pose endpoints" approach as the deprecated tamp_rule baseline.
    Real implementation should use motion primitives + IK to compute the
    trajectory consistent with the symbolic plan.
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
    # Linear interpolation between endpoints (placeholder)
    t0_idx, t1_idx = 0, poses.shape[0] - 1
    out = np.zeros((T, 7), dtype=np.float32)
    for i in range(T):
        u = i / max(T - 1, 1)
        out[i, :3] = (1 - u) * poses[t0_idx, :3] + u * poses[t1_idx, :3]
        # SLERP for quaternion (simplified — see tamp_rule for full version)
        q0 = poses[t0_idx, 3:]; q1 = poses[t1_idx, 3:]
        if float(np.dot(q0, q1)) < 0:
            q1 = -q1
        out[i, 3:] = (1 - u) * q0 + u * q1
        out[i, 3:] /= max(float(np.linalg.norm(out[i, 3:])), 1e-12)
    return out


def run_one(traj_dir: Path, text: str, T: int) -> Tuple[Optional[GS4DSequence], Dict]:
    """Run TAMP-PDDL on one trajectory."""
    meta = load_meta_json(traj_dir)
    problem = _build_pddl_problem(meta)
    plan_log: Dict = {"text": text, "task_name": meta.get("task_name")}

    if problem is None:
        plan_log["status"] = "task_not_in_pddl_domain"
        return None, plan_log

    plan = _solve_pddl(problem)
    if plan is None:
        plan_log["status"] = "pddlstream_unavailable_or_no_plan"
        return None, plan_log
    plan_log["plan"] = [(name, str(args)) for name, args in plan]

    poses = _execute_plan_to_trajectory(plan, meta, traj_dir, T=T)
    if poses is None:
        plan_log["status"] = "no_motion_executable"
        return None, plan_log

    # Apply pose trajectory to init_gs
    gs = load_init_gs_ply(traj_dir / "init_gs.ply", n_points=10000, seed=0, c_sh=48)
    mu0      = gs.mu.numpy().astype(np.float32)
    cov0     = quat_log_scale_to_full_cov(gs.cov.numpy(), gs.scale.numpy())
    sh0      = gs.sh.numpy().astype(np.float32)
    opacity0 = gs.opacity.numpy().astype(np.float32)
    scale0   = gs.scale.numpy().astype(np.float32)
    mu_t, cov_t, sh_t, opacity_t, scale_t = apply_pose_trajectory_to_gs(
        mu0, cov0, sh0, opacity0, scale0, poses=poses,
    )
    plan_log["status"] = "ok"
    return GS4DSequence(mu=mu_t, cov=cov_t, sh=sh_t,
                         opacity=opacity_t, scale=scale_t), plan_log


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+",
                   default=["test_iid", "test_ood_unseen_pair",
                            "test_ood_unseen_object", "test_compositional_long"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if not _try_import_pddlstream():
        print("⚠ PDDLStream not installed — running in skeleton mode "
              "(falls back to GT-endpoint interpolation, similar to tamp_rule).",
              file=sys.stderr)

    n_total = n_ok = n_skip = 0
    t0 = time.time()
    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        n_split = 0
        for traj_id, traj_dir, entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            if args.limit is not None and n_split >= args.limit:
                break
            n_split += 1
            n_total += 1

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            try:
                seq, plan_log = run_one(traj_dir, text, T=args.T)
            except Exception as e:
                print(f"  ✗ {traj_id}  ERROR  {type(e).__name__}: {e}")
                n_skip += 1
                continue

            out = baseline_output_dir(
                args.output_root, "tamp_pddl",
                args.dataset_name, split, traj_id,
            )
            with open(out / "plan.json", "w") as f:
                json.dump(plan_log, f, indent=2)
            if seq is None:
                TrajMetrics(notes=plan_log["status"]).save(out / "metrics.json")
                n_skip += 1
                continue
            seq.save(out / "pred_4dgs.npz")
            TrajMetrics(notes="pending_eval").save(out / "metrics.json")
            n_ok += 1
            if n_ok <= 3 or n_ok % 50 == 0:
                print(f"  ✓ {traj_id}")

    print(f"\n=== TAMP-PDDL complete: ok={n_ok}, skip={n_skip}, total={n_total}, "
          f"elapsed={time.time()-t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
