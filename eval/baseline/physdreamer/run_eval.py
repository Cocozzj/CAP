"""Invoke PhysDreamer on every prepared trajectory, collect outputs into
the unified baseline format.

Workflow:
  1. ``convert_data.py`` writes ``physdreamer_config.json`` + ``init_gs.ply``
     under ``runs/baselines/physdreamer/<dataset>/<split>/<traj_id>/``.
  2. This script iterates each, calls PhysDreamer's inference command with
     the config + ply, and writes the output into ``raw/`` next to the config.
  3. ``parse_outputs.py`` translates the raw output → ``pred_4dgs.npz``.

PhysDreamer's exact CLI may differ between versions.  See the
``--probe`` flag below to print the constructed command without running.

Usage:

    # Set environment variable once:
    export PHYSDREAMER_REPO=~/PhysDreamer

    # Then iterate all prepared trajectories:
    python -m eval.baseline.physdreamer.run_eval \\
        --output-root runs/baselines \\
        --dataset-name dataset_a \\
        --splits test_iid \\
        --physdreamer-repo $PHYSDREAMER_REPO

    # Dry-run (just print what would be invoked):
    python -m eval.baseline.physdreamer.run_eval ... --probe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..common import TrajMetrics
from .parse_outputs import parse_physdreamer_output


# ──────────────────────────────────────────────────────────────────────
# Discover PhysDreamer's inference entrypoint
# ──────────────────────────────────────────────────────────────────────

def _discover_inference_script(repo: Path) -> Optional[Path]:
    """Try common script names for PhysDreamer's inference entry."""
    candidates = [
        "inference.py",
        "run_inference.py",
        "scripts/inference.py",
        "scripts/run.py",
        "demo.py",
    ]
    for c in candidates:
        p = repo / c
        if p.exists():
            return p
    return None


def _build_cmd(
    script: Path,
    config_path: Path,
    init_gs_path: Path,
    out_dir: Path,
    extra_args: List[str],
) -> List[str]:
    """Default CLI assumption — adjust to PhysDreamer's actual args."""
    return [
        "python", str(script),
        "--config",      str(config_path),
        "--model_path",  str(init_gs_path),
        "--output_path", str(out_dir),
        *extra_args,
    ]


def _run_one(
    script:        Path,
    config_path:   Path,
    init_gs_path:  Path,
    out_dir:       Path,
    repo_cwd:      Path,
    timeout_secs:  int,
    extra_args:    List[str],
    probe:         bool = False,
) -> Tuple[bool, str]:
    """Invoke PhysDreamer once on a single trajectory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(script, config_path, init_gs_path, out_dir, extra_args)

    if probe:
        print("  [probe] " + " ".join(str(c) for c in cmd))
        return True, "probe (not executed)"

    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs,
            cwd=str(repo_cwd),
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_secs}s"
    except FileNotFoundError as e:
        return False, f"script not found: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if r.returncode != 0:
        err = (r.stderr or "")[-400:]
        return False, f"returncode={r.returncode}  stderr_tail={err!r}"
    return True, "ok"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root",     default="runs/baselines")
    p.add_argument("--dataset-name",    default="dataset_a")
    p.add_argument("--splits",          nargs="+", default=None,
                   help="if None, run on all splits found")
    p.add_argument("--physdreamer-repo", required=False,
                   default=os.environ.get("PHYSDREAMER_REPO", ""),
                   help="path to cloned PhysDreamer repo (or set $PHYSDREAMER_REPO)")
    p.add_argument("--script", default=None,
                   help="explicit path to inference.py (overrides auto-discovery)")
    p.add_argument("--extra-args", nargs="*", default=[],
                   help="extra CLI args to pass through to PhysDreamer's inference")
    p.add_argument("--timeout", type=int, default=900,
                   help="per-trajectory timeout (s); diffusion is slow, default 15min")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip trajectories that already have pred_4dgs.npz")
    p.add_argument("--probe", action="store_true",
                   help="dry-run: print constructed commands, do not execute")
    args = p.parse_args(argv)

    # Locate PhysDreamer
    if not args.physdreamer_repo:
        print("✗ --physdreamer-repo required (or set $PHYSDREAMER_REPO)",
              file=sys.stderr)
        return 1
    repo = Path(args.physdreamer_repo).expanduser().resolve()
    if not repo.exists():
        print(f"✗ PhysDreamer repo not found at {repo}", file=sys.stderr)
        return 1

    script = Path(args.script).expanduser().resolve() if args.script else \
             _discover_inference_script(repo)
    if script is None or not script.exists():
        print(f"✗ could not find PhysDreamer's inference script under {repo}",
              file=sys.stderr)
        print(f"  use --script to specify it explicitly", file=sys.stderr)
        return 1
    print(f"⏬ using PhysDreamer script: {script}")

    base = Path(args.output_root) / "physdreamer" / args.dataset_name
    if not base.exists():
        print(f"✗ no convert_data outputs at {base}", file=sys.stderr)
        print(f"  run convert_data.py first to prepare per-trajectory configs.",
              file=sys.stderr)
        return 1

    splits = args.splits or [d.name for d in sorted(base.iterdir()) if d.is_dir()]

    n_total = n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for split in splits:
        split_dir = base / split
        if not split_dir.exists():
            continue
        print(f"\n=== Split: {split} ===")
        for traj_dir in sorted(split_dir.iterdir()):
            if not traj_dir.is_dir():
                continue
            if args.limit is not None and n_total >= args.limit:
                break
            n_total += 1

            cfg_path  = traj_dir / "physdreamer_config.json"
            ply_path  = traj_dir / "init_gs.ply"
            pred_path = traj_dir / "pred_4dgs.npz"

            if not cfg_path.exists() or not ply_path.exists():
                n_skip += 1
                continue
            if args.skip_existing and pred_path.exists():
                n_skip += 1
                continue

            raw_dir = traj_dir / "raw"
            success, msg = _run_one(
                script, cfg_path, ply_path, raw_dir,
                repo_cwd=repo, timeout_secs=args.timeout,
                extra_args=args.extra_args, probe=args.probe,
            )
            if not success:
                TrajMetrics(notes=f"physdreamer_failed: {msg[:200]}").save(
                    traj_dir / "metrics.json")
                if n_fail < 3 or n_fail % 10 == 0:
                    print(f"  ✗ {traj_dir.name}  {msg}")
                n_fail += 1
                continue

            if args.probe:
                # Don't try to parse outputs in probe mode
                n_ok += 1
                continue

            # Parse PhysDreamer's raw output → GS4DSequence
            seq = parse_physdreamer_output(raw_dir, init_ply=ply_path)
            if seq is None:
                TrajMetrics(notes="physdreamer_output_parse_failed").save(
                    traj_dir / "metrics.json")
                n_fail += 1
                if n_fail < 3:
                    print(f"  ✗ {traj_dir.name}  ran but couldn't parse outputs in {raw_dir}")
                    print(f"    Inspect that directory and adjust parse_outputs.py")
                continue

            seq.save(pred_path)
            TrajMetrics(notes="pending_eval").save(traj_dir / "metrics.json")
            n_ok += 1
            if n_ok <= 3 or n_ok % 50 == 0:
                print(f"  ✓ {traj_dir.name}  T={seq.T} N={seq.N}")

    print(f"\n=== PhysDreamer complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skip}")
    print(f"  failed:  {n_fail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
