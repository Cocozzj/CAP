"""Generate per-trajectory PhysGaussian configs from a dataset split.

For each trajectory in the split:
  - locate init_gs.ply (PhysGaussian's input)
  - read GT ρ from physics.json if available, otherwise use a category-based fallback
  - write a PhysGaussian config JSON to:
      runs/baselines/physgaussian/<dataset>/<split>/<traj_id>/physgs_config.json

Then `run_eval.py` reads each config, invokes PhysGaussian, and writes outputs.

Usage:

    python -m eval.baseline.physgaussian.convert_data \\
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
from pathlib import Path
from typing import List

from ..common import (
    baseline_output_dir,
    iter_split_entries,
    load_physics_json,
)
from .rho_to_config import default_rho_for_partnet_object, rho_to_physgaussian_config


def _extract_rho_from_physics_json(p: dict) -> list[float] | None:
    """Build the 9-tuple ρ from our actual physics.json schema.

    Confirmed schema (from CAP/dataset/dataset_a/data/<traj_id>/physics.json):
        {"friction": float, "damping": float}

    The remaining 7 slots (E, ν, ρ_m, F[3], dt) are NOT in physics.json — we
    use category-aware defaults (PartNet objects are mostly metallic rigid).
    """
    if not isinstance(p, dict):
        return None
    if "friction" not in p or "damping" not in p:
        return None
    # Default: metal-like rigid + gravity + 30fps timestep
    E, nu, rho_m = 2.0e11, 0.30, 7800.0
    fx, fy, fz   = 0.0, 0.0, -9.81
    dt           = 1.0 / 30.0
    return [
        E, nu, rho_m,
        fx, fy, fz,
        float(p["friction"]), float(p["damping"]), dt,
    ]


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+", default=["test_iid"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args(argv)

    n_total = 0
    n_with_gt_rho = 0
    n_fallback    = 0

    for split in args.splits:
        for traj_id, traj_dir, entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            n_total += 1
            phy = load_physics_json(traj_dir)
            rho = _extract_rho_from_physics_json(phy) if phy else None

            if rho is None:
                rho = list(default_rho_for_partnet_object(entry.get("obj_category")))
                n_fallback += 1
            else:
                n_with_gt_rho += 1

            cfg = rho_to_physgaussian_config(rho, n_frames=args.T, fps=args.fps)

            # Pointer to init_gs.ply (PhysGaussian's input)
            cfg["model_path"] = str((traj_dir / "init_gs.ply").resolve())
            cfg["traj_id"]    = traj_id
            cfg["dataset"]    = args.dataset_name
            cfg["split"]      = split

            out_dir = baseline_output_dir(
                args.output_root, "physgaussian",
                args.dataset_name, split, traj_id,
            )
            with open(out_dir / "physgs_config.json", "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"Wrote {n_total} configs ({n_with_gt_rho} from physics.json, "
          f"{n_fallback} from category fallback).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
