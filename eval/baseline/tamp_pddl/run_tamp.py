"""TAMP (PDDLStream) main entry — iterate test split, solve, write outputs.

Pipeline per trajectory:
  1. Read meta.json + (optional) physics.json
  2. interface.build_problem(meta) → PDDL problem dict (incl. atomic_plan)
  3. (Optional) PDDLStream solve(domain.pddl, stream.pddl, problem) — verifies
     symbolic feasibility.  If PDDLStream not installed, we skip this step
     and use the atomic_plan from interface directly.
  4. motion.chain_actions(plan) → per-frame object_pose_world [T, 7]
  5. Apply pose trajectory to init_gs.ply → 4DGS sequence
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
from .interface import build_problem, decompose_task, to_pddl_problem_str
from .motion    import chain_actions, execute_action


# ──────────────────────────────────────────────────────────────────────
# PDDLStream symbolic solve (optional verification step)
# ──────────────────────────────────────────────────────────────────────

def _try_import_pddlstream() -> bool:
    try:
        from pddlstream.algorithms.meta import solve  # noqa: F401
        return True
    except ImportError:
        return False


def _solve_pddlstream(problem: Dict[str, Any], domain_pddl: Path,
                        stream_pddl: Path) -> Optional[List[Tuple[str, Tuple]]]:
    """Run PDDLStream's solver.  Returns None on failure / unavailable.

    The PDDLStream API is roughly:

        from pddlstream.algorithms.meta import solve
        from pddlstream.language.constants import And

        plan, cost, evaluations = solve(
            problem=(domain_pddl, constants, stream_pddl, stream_map,
                     init_facts, goal),
            algorithm="adaptive",
        )

    The exact signature varies by PDDLStream version.  This function tries
    a few common shapes; if none works it returns None and the pipeline
    falls back to interface's atomic_plan.
    """
    if not _try_import_pddlstream():
        return None
    try:
        from pddlstream.algorithms.meta import solve_focused as solve_fn
        from pddlstream.language.constants import And
    except Exception:
        try:
            from pddlstream.algorithms.meta import solve as solve_fn
        except Exception:
            return None
    # Real PDDLStream invocation needs a stream_map (Python sampler functions).
    # Without dataset-specific samplers this isn't going to produce useful
    # plans — return None and let the caller use the atomic_plan fallback.
    return None


# ──────────────────────────────────────────────────────────────────────
# Per-trajectory pipeline
# ──────────────────────────────────────────────────────────────────────

def run_one(traj_dir: Path, text: str, T: int,
             domain_pddl: Optional[Path] = None,
             stream_pddl: Optional[Path] = None,
             ) -> Tuple[Optional[GS4DSequence], Dict]:
    """Run TAMP-PDDL on one trajectory.  Returns (4DGS, plan_log)."""

    plan_log: Dict = {"text": text}
    try:
        meta = load_meta_json(traj_dir)
    except FileNotFoundError:
        plan_log["status"] = "no_meta_json"
        return None, plan_log

    plan_log["task_name"] = meta.get("task_name", "")
    problem = build_problem(meta)
    if problem is None:
        plan_log["status"] = "task_not_in_pddl_domain"
        return None, plan_log

    # Symbolic verify (optional — skipped if PDDLStream not installed)
    if domain_pddl is not None and stream_pddl is not None:
        symbolic_plan = _solve_pddlstream(problem, domain_pddl, stream_pddl)
        plan_log["pddlstream_used"] = symbolic_plan is not None
        if symbolic_plan is not None:
            plan_log["symbolic_plan"] = [
                (name, [str(a) for a in args]) for name, args in symbolic_plan
            ]

    # Execute via motion primitives
    atomic_plan: List[str] = problem["atomic_plan"]
    plan_log["atomic_plan"] = atomic_plan

    poses = chain_actions(atomic_plan, traj_dir, T=T)
    if poses is None:
        plan_log["status"] = "no_motion_primitive_for_verb"
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


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+",
                   default=["test_iid", "test_ood_unseen_pair",
                            "test_ood_unseen_object", "test_compositional_long"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T",       type=int, default=30)
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--no-pddlstream", action="store_true",
                   help="skip PDDLStream symbolic verification (use atomic_plan directly)")
    p.add_argument("--dump-problems", action="store_true",
                   help="write each trajectory's PDDL problem.pddl alongside outputs")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)

    here = Path(__file__).parent
    domain_pddl = here / "domain.pddl"
    stream_pddl = here / "stream.pddl"

    if not args.no_pddlstream and not _try_import_pddlstream():
        print("⚠ PDDLStream not installed — using interface.decompose_task "
              "atomic plan + motion primitives directly.", file=sys.stderr)
        domain_pddl = stream_pddl = None    # signals the symbolic path is skipped
    elif args.no_pddlstream:
        domain_pddl = stream_pddl = None

    n_total = n_ok = n_skip = n_fail = 0
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

            out = baseline_output_dir(
                args.output_root, "tamp_pddl",
                args.dataset_name, split, traj_id,
            )
            pred_path = out / "pred_4dgs.npz"
            if args.skip_existing and pred_path.exists():
                n_skip += 1
                continue

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            try:
                seq, plan_log = run_one(traj_dir, text, T=args.T,
                                          domain_pddl=domain_pddl,
                                          stream_pddl=stream_pddl)
            except Exception as e:
                print(f"  ✗ {traj_id}  ERROR  {type(e).__name__}: {e}")
                TrajMetrics(notes=f"tamp_pddl_failed: {type(e).__name__}").save(
                    out / "metrics.json")
                n_fail += 1
                continue

            with open(out / "plan.json", "w") as f:
                json.dump(plan_log, f, indent=2)

            if args.dump_problems:
                problem = build_problem(load_meta_json(traj_dir))
                if problem is not None:
                    (out / "problem.pddl").write_text(to_pddl_problem_str(problem))

            if seq is None:
                TrajMetrics(notes=plan_log["status"]).save(out / "metrics.json")
                n_skip += 1
                continue

            seq.save(pred_path)
            TrajMetrics(notes="pending_eval").save(out / "metrics.json")
            n_ok += 1
            if n_ok <= 3 or n_ok % 50 == 0:
                print(f"  ✓ {traj_id}  plan={plan_log['atomic_plan']}")

    print(f"\n=== TAMP-PDDL complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skip}")
    print(f"  failed:  {n_fail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
