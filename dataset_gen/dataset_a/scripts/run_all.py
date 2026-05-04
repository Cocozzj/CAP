"""Run the full Dataset-A pipeline end to end.

Convenience wrapper around scripts 01-05.

Pipeline order (NEW, post-pivot):
    1. Filter PartNet-Mobility objects
    2. Generate trajectory plans + run physics in SAPIEN
    3. Render multi-view RGB videos
    4. From each trajectory's first frame, derive static 3DGS
    5. Build splits + manifest
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent  # dataset/dataset_a/


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}\n")
    res = subprocess.run(cmd, cwd=PROJ)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partnet_root", required=True,
                    help="Path to PartNet-Mobility extraction (contains dataset/<obj_id>/...)")
    ap.add_argument("--out", default="outputs/")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--gs_backend", default=None,
                    help="Override gs.backend: mvsplat | gsplat | hybrid")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--skip_steps", default="",
                    help="Comma-separated step numbers to skip, e.g. '4' to skip 3DGS")
    args = ap.parse_args()

    out = Path(args.out)
    skips = set(args.skip_steps.split(",")) if args.skip_steps else set()
    py = sys.executable

    # ---- Step 1: filter objects
    if "1" not in skips:
        run([py, "scripts/01_filter_objects.py",
             "--partnet_root", args.partnet_root,
             "--config", args.config,
             "--out", str(out / "object_list.json")])

    # ---- Step 2: generate trajectory plans (CPU-bound physics)
    if "2" not in skips:
        run([py, "scripts/02_generate_trajectories.py",
             "--object_list", str(out / "object_list.json"),
             "--config", args.config,
             "--out", str(out / "trajectories.json")])

    # ---- Step 3: render multi-view videos (GPU-bound)
    if "3" not in skips:
        run([py, "scripts/03_render_multiview.py",
             "--trajectories", str(out / "trajectories.json"),
             "--object_list", str(out / "object_list.json"),
             "--config", args.config,
             "--num_workers", str(args.num_workers),
             "--out", str(out / "data")])

    # ---- Step 4: per-trajectory first-frame 3DGS via configured backend
    if "4" not in skips:
        cmd = [py, "scripts/04_init_gs_from_first_frame.py",
               "--data_dir", str(out / "data"),
               "--config", args.config,
               "--num_workers", str(args.num_workers)]
        if args.gs_backend:
            cmd += ["--backend", args.gs_backend]
        run(cmd)

    # ---- Step 5: split + manifest
    if "5" not in skips:
        run([py, "scripts/05_split_and_pack.py",
             "--trajectories", str(out / "trajectories.json"),
             "--data_dir", str(out / "data"),
             "--config", args.config,
             "--out_splits", str(out / "splits.json"),
             "--out_manifest", str(out / "manifest.json")])

    print("\nALL DONE.")


if __name__ == "__main__":
    main()
