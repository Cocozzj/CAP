"""Generate per-trajectory PhysDreamer input files.

For each trajectory in the split:
  - Locate init_gs.ply (PhysDreamer's static scene input)
  - Read GT ρ from physics.json if available, else use category fallback
  - Write a PhysDreamer config JSON to:
      runs/baselines/physdreamer/<dataset>/<split>/<traj_id>/physdreamer_config.json
  - Symlink (or copy) init_gs.ply for convenience

Then ``run_eval.py`` reads each config + .ply, invokes PhysDreamer, and
collects outputs.

Usage:

    python -m eval.baseline.physdreamer.convert_data \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines \\
        --T 30 --fps 30
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

from ..common import (
    baseline_output_dir,
    iter_split_entries,
    load_physics_json,
)
from .rho_to_config import default_rho_for_partnet_object, rho_to_physdreamer_config


# Same friction+damping → ρ extraction as the PhysGaussian wrapper.
def _extract_rho_from_physics_json(p: dict, obj_category: str) -> Optional[List[float]]:
    """Build the 9-tuple ρ from physics.json (only friction + damping there).

    Confirmed schema (from CAP/dataset/dataset_a/data/<traj_id>/physics.json):
        {"friction": float, "damping": float}

    The remaining 7 slots come from a category-aware default.
    """
    if not isinstance(p, dict):
        return None
    if "friction" not in p or "damping" not in p:
        return None
    base = list(default_rho_for_partnet_object(obj_category))
    base[6] = float(p["friction"])     # μ
    base[7] = float(p["damping"])      # damping
    return base


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+", default=["test_iid"])
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T",   type=int, default=30,
                   help="number of output frames (PhysDreamer simulates "
                        "T/fps seconds = duration)")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--n-substeps", type=int, default=100,
                   help="MPM substeps per output frame; higher = more stable, "
                        "slower; PhysDreamer default ~100")
    p.add_argument("--copy-ply", action="store_true",
                   help="copy init_gs.ply into output dir (vs symlink). Use this "
                        "if PhysDreamer's filesystem can't follow symlinks.")
    args = p.parse_args(argv)

    n_total = 0
    n_with_gt_rho = 0
    n_fallback    = 0

    for split in args.splits:
        for traj_id, traj_dir, entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            n_total += 1
            obj_cat = entry.get("obj_category", "")

            phys = load_physics_json(traj_dir)
            rho = _extract_rho_from_physics_json(phys, obj_cat) if phys else None
            if rho is None:
                rho = list(default_rho_for_partnet_object(obj_cat))
                n_fallback += 1
            else:
                n_with_gt_rho += 1

            duration = args.T / float(args.fps)
            cfg = rho_to_physdreamer_config(
                rho, duration_secs=duration, fps=args.fps,
                n_substeps=args.n_substeps,
            )
            cfg["traj_id"]      = traj_id
            cfg["dataset"]      = args.dataset_name
            cfg["split"]        = split
            cfg["output_frames"]= args.T

            out_dir = baseline_output_dir(
                args.output_root, "physdreamer",
                args.dataset_name, split, traj_id,
            )
            (out_dir / "raw").mkdir(parents=True, exist_ok=True)

            # Persist config
            cfg["model_path"] = str((out_dir / "init_gs.ply").resolve())
            with open(out_dir / "physdreamer_config.json", "w") as f:
                json.dump(cfg, f, indent=2)

            # Stage init_gs.ply next to the config
            src = traj_dir / "init_gs.ply"
            dst = out_dir / "init_gs.ply"
            if dst.exists():
                dst.unlink()
            if args.copy_ply:
                shutil.copy(src, dst)
            else:
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    # symlink not allowed (e.g. some Windows mounts) → fall back to copy
                    shutil.copy(src, dst)

    print(f"\n=== PhysDreamer convert_data complete ===")
    print(f"  total entries:               {n_total}")
    print(f"  with GT physics.json (μ, ν): {n_with_gt_rho}")
    print(f"  with category fallback:      {n_fallback}")
    print(f"\nNext step: run inference")
    print(f"  python -m eval.baseline.physdreamer.run_eval "
          f"--output-root {args.output_root} --physdreamer-repo $PHYSDREAMER_REPO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
