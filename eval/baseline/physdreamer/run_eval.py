"""Invoke PhysDreamer on each test trajectory and collect outputs.

SKELETON — adjust to PhysDreamer's actual CLI / output format after cloning
their repo.  See ../README.md for setup.

Usage:

    python -m eval.baseline.physdreamer.run_eval \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines \\
        --physdreamer-repo $PHYSDREAMER_REPO
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from dataload.text import task_to_text

from ..common import (
    GS4DSequence,
    TrajMetrics,
    baseline_output_dir,
    iter_split_entries,
)


def _run_physdreamer_one(
    init_gs_path: Path,
    text:         str,
    out_dir:      Path,
    physdreamer_repo: Path,
    timeout_secs: int = 600,
) -> tuple[bool, str]:
    """Call PhysDreamer's inference script on a single trajectory.

    TODO: replace this with the actual command after inspecting their repo:
      python <repo>/inference.py --gs <init_gs.ply> --prompt <text> --output <dir>
    """
    cmd = [
        "python", str(physdreamer_repo / "inference.py"),
        "--gs",     str(init_gs_path),
        "--prompt", text,
        "--output", str(out_dir),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                            timeout=timeout_secs, cwd=str(physdreamer_repo))
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_secs}s"
    except FileNotFoundError as e:
        return False, f"physdreamer script not found: {e}"

    if r.returncode != 0:
        return False, f"returncode={r.returncode}  stderr_tail={(r.stderr or '')[-300:]!r}"
    return True, "ok"


def _collect_physdreamer_output(out_dir: Path) -> Optional[GS4DSequence]:
    """Read PhysDreamer's per-trajectory output and convert to our GS4DSequence.

    TODO: implement after inspecting PhysDreamer's actual output schema.
    Typical schemas:
      - Per-frame .ply files in `frames/`
      - A single .npz with stacked Gaussian arrays
      - A rendered video (no 4DGS; in that case we'd need a fallback)
    """
    # Try common output formats
    npz_files = list(out_dir.glob("*.npz"))
    if npz_files:
        try:
            z = np.load(npz_files[0])
            return GS4DSequence(
                mu=z["mu"], cov=z["cov"], sh=z["sh"],
                opacity=z["opacity"], scale=z["scale"],
            )
        except KeyError:
            pass
    return None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+", default=["test_iid"])
    p.add_argument("--output-root",     default="runs/baselines")
    p.add_argument("--dataset-name",    default="dataset_a")
    p.add_argument("--physdreamer-repo", required=True,
                   help="path to cloned PhysDreamer repo (or set $PHYSDREAMER_REPO)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--limit",   type=int, default=None)
    args = p.parse_args(argv)

    pd_repo = Path(args.physdreamer_repo).expanduser().resolve()
    if not (pd_repo / "inference.py").exists():
        print(f"⚠ PhysDreamer inference.py not at {pd_repo} — paths in run_eval.py "
              f"may need adjustment.", file=sys.stderr)

    n_total = n_ok = n_fail = 0
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

            text     = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            init_gs  = traj_dir / "init_gs.ply"
            out_dir  = baseline_output_dir(
                args.output_root, "physdreamer",
                args.dataset_name, split, traj_id,
            )

            success, msg = _run_physdreamer_one(
                init_gs, text, out_dir / "raw", pd_repo, timeout_secs=args.timeout,
            )
            if not success:
                TrajMetrics(notes=f"physdreamer_failed: {msg}").save(out_dir / "metrics.json")
                n_fail += 1
                continue

            seq = _collect_physdreamer_output(out_dir / "raw")
            if seq is None:
                TrajMetrics(notes="physdreamer_output_parse_failed").save(out_dir / "metrics.json")
                n_fail += 1
                continue

            seq.save(out_dir / "pred_4dgs.npz")
            TrajMetrics(notes="pending_eval").save(out_dir / "metrics.json")
            n_ok += 1
            if n_ok <= 3 or n_ok % 50 == 0:
                print(f"  ✓ {traj_id}")

    print(f"\n=== PhysDreamer complete: ok={n_ok}, failed={n_fail}, "
          f"total={n_total}, elapsed={time.time()-t0:.1f}s ===")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
