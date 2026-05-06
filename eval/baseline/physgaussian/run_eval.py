"""Invoke PhysGaussian on every trajectory's pre-built config.

Workflow:
  1. ``convert_data.py`` already wrote one ``physgs_config.json`` per trajectory.
  2. This script iterates over those configs, calls PhysGaussian's
     ``gs_simulation.py`` with each one, and saves the resulting deformed
     Gaussian sequence to ``pred_4dgs.npz``.
  3. PhysGaussian's exact CLI / output schema may differ from this stub.
     See the TODO markers and adjust against the version you cloned.

Usage:

    python -m eval.baseline.physgaussian.run_eval \\
        --output-root runs/baselines \\
        --physgs-repo $PHYSGAUSSIAN_REPO
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from ..common import GS4DSequence, TrajMetrics


def _run_physgaussian_one(
    cfg_path: Path,
    output_dir: Path,
    physgs_repo: Path,
    timeout_secs: int = 600,
) -> tuple[bool, str]:
    """Invoke PhysGaussian's ``gs_simulation.py`` on one config.

    PhysGaussian's CLI (verify against your cloned version):
        python gs_simulation.py \\
            --config <cfg.json> \\
            --output_path <out_dir>

    Returns (success, message).  On failure, message is the exception/error
    string for logging.

    PYTHONPATH:  ``utils.sh_utils`` lives in PhysGaussian's
    ``gaussian-splatting/`` submodule (NOT in the top-level ``utils/``).
    We prepend it to PYTHONPATH so ``gs_simulation.py``'s
    ``from utils.sh_utils import eval_sh`` resolves.
    """
    import os
    cmd = [
        "python", str(physgs_repo / "gs_simulation.py"),
        "--config",     str(cfg_path),
        "--output_path", str(output_dir),
    ]
    env = dict(os.environ)
    extra_paths = [
        str(physgs_repo / "gaussian-splatting"),     # for utils.sh_utils, scene/, etc.
        str(physgs_repo),                             # for top-level imports
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs,
            cwd=str(physgs_repo), env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_secs}s"
    except FileNotFoundError as e:
        return False, f"physgaussian script not found: {e}"

    if r.returncode != 0:
        # Common failures: solver divergence on thin objects, OOM on dense scenes
        err = (r.stderr or "")[-500:]
        return False, f"returncode={r.returncode}  stderr_tail={err!r}"
    return True, "ok"


def _collect_physgaussian_outputs(physgs_out_dir: Path) -> GS4DSequence | None:
    """Read whatever PhysGaussian wrote and convert to our GS4DSequence.

    PhysGaussian's exact output schema (verify against your version):
      - ``frames/frame_000.ply, frame_001.ply, ...`` per-timestep PLYs, OR
      - ``simulation.npz`` with arrays ``mu, cov, ...``

    This stub tries both.  Adjust to whatever your version actually writes.
    """
    # Schema 1: per-frame PLY files
    ply_files = sorted(physgs_out_dir.glob("frames/frame_*.ply"))
    if ply_files:
        try:
            from dataload.common import load_init_gs_ply
        except ImportError:
            return None

        T = len(ply_files)
        # Read first to learn N
        first = load_init_gs_ply(ply_files[0], n_points=10000, seed=0, c_sh=48)
        N = int(first.mu.shape[0])

        mu = np.zeros((T, N, 3), dtype=np.float32)
        for t, p in enumerate(ply_files):
            gs = load_init_gs_ply(p, n_points=N, seed=0, c_sh=48)
            mu[t] = gs.mu.numpy()
        # Reuse first frame's cov/sh/opacity/scale as broadcast (PhysGaussian
        # may not write them per-frame; verify in your version)
        cov0     = first.cov.numpy()  # placeholder, may need conversion
        sh0      = first.sh.numpy()
        opacity0 = first.opacity.numpy()
        scale0   = first.scale.numpy()

        # Build a 3x3 cov from quat (caller's job) — for now broadcast
        cov_full = np.eye(3, dtype=np.float32)[None, None].repeat(T, axis=0).repeat(N, axis=1)

        return GS4DSequence(
            mu=mu, cov=cov_full,
            sh=np.broadcast_to(sh0[None],      (T,) + sh0.shape).copy(),
            opacity=np.broadcast_to(opacity0[None], (T,) + opacity0.shape).copy(),
            scale=np.broadcast_to(scale0[None],   (T,) + scale0.shape).copy(),
        )

    # Schema 2: single npz
    npz_files = list(physgs_out_dir.glob("*.npz"))
    if npz_files:
        z = np.load(npz_files[0])
        # Expected keys: mu, cov, sh, opacity, scale  (verify schema)
        try:
            return GS4DSequence(
                mu=z["mu"], cov=z["cov"], sh=z["sh"],
                opacity=z["opacity"], scale=z["scale"],
            )
        except KeyError:
            return None

    return None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--splits", nargs="+", default=None,
                   help="if None, run on all splits found under output_root/physgaussian/<dataset>/")
    p.add_argument("--physgs-repo", required=True,
                   help="path to cloned PhysGaussian repo (set $PHYSGAUSSIAN_REPO)")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-trajectory timeout (PhysGaussian sims can hang on degenerate input)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    physgs_repo = Path(args.physgs_repo).expanduser().resolve()
    if not (physgs_repo / "gs_simulation.py").exists():
        print(f"✗ PhysGaussian script not found at {physgs_repo}", file=sys.stderr)
        print(f"  set --physgs-repo or $PHYSGAUSSIAN_REPO to the cloned repo path",
              file=sys.stderr)
        return 1

    base = Path(args.output_root) / "physgaussian" / args.dataset_name
    if args.splits is None:
        splits = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
    else:
        splits = args.splits

    total = ok = failed = 0
    t0 = time.time()
    for split in splits:
        split_dir = base / split
        traj_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        for i, traj_dir in enumerate(traj_dirs):
            if args.limit is not None and total >= args.limit:
                break
            total += 1
            cfg_path = traj_dir / "physgs_config.json"
            if not cfg_path.exists():
                print(f"  ⊘ {traj_dir.name}: no physgs_config.json (run convert_data first)")
                failed += 1
                continue

            # PhysGaussian writes its outputs into a subdir of traj_dir
            physgs_out = traj_dir / "physgs_raw"
            physgs_out.mkdir(exist_ok=True)

            success, msg = _run_physgaussian_one(
                cfg_path, physgs_out, physgs_repo, timeout_secs=args.timeout,
            )
            if not success:
                TrajMetrics(notes=f"physgs_failed: {msg}").save(traj_dir / "metrics.json")
                failed += 1
                if failed <= 3:
                    print(f"  ✗ {traj_dir.name}  {msg}")
                continue

            # Convert PhysGaussian raw output → our unified GS4DSequence format
            seq = _collect_physgaussian_outputs(physgs_out)
            if seq is None:
                TrajMetrics(notes="physgs_output_parse_failed").save(traj_dir / "metrics.json")
                failed += 1
                continue

            seq.save(traj_dir / "pred_4dgs.npz")
            TrajMetrics(notes="pending_eval").save(traj_dir / "metrics.json")
            ok += 1
            if ok <= 3 or ok % 50 == 0:
                print(f"  ✓ {traj_dir.name}  T={seq.T} N={seq.N}")

    dt = time.time() - t0
    print(f"\n=== PhysGaussian complete ===")
    print(f"  total:   {total}")
    print(f"  ok:      {ok}")
    print(f"  failed:  {failed}")
    print(f"  elapsed: {dt:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
