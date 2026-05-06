"""Generate per-trajectory SVD input metadata.

For each trajectory in the split:
  - Locate cam0.mp4 (we read its first frame as SVD's image input)
  - Write a tiny svd_config.json recording the source video path
    and target generation length

Output layout:
    runs/baselines/svd/<dataset>/<split>/<traj_id>/svd_config.json

Then ``run_eval.py`` reads each config, loads the first frame, runs SVD,
and writes ``pred_render.mp4``.

Usage:

    python -m eval.baseline.svd.convert_data \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines

This is much lighter than the PhysGaussian convert_data because SVD takes
no physics parameters — it's a generic image-to-video model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from ..common import baseline_output_dir, iter_split_entries


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+", default=["test_iid"])
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    # SVD-XT defaults: 25 frames at 6 fps = 4.16s clips.  Our GT trajs are
    # 30 frames at 30fps = 1s; we generate 25 frames (SVD's native) and
    # render_metrics.py downsamples to T frames at compare time.
    p.add_argument("--n-frames", type=int, default=25,
                   help="number of frames SVD should generate (svd-xt: 25)")
    p.add_argument("--fps",      type=int, default=6,
                   help="output fps tag (cosmetic; metric path resamples)")
    p.add_argument("--height",   type=int, default=576,
                   help="SVD output height (svd-xt native: 576)")
    p.add_argument("--width",    type=int, default=1024,
                   help="SVD output width (svd-xt native: 1024)")
    args = p.parse_args(argv)

    n_total = n_skipped = 0
    for split in args.splits:
        for traj_id, traj_dir, _entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            n_total += 1
            cam0 = Path(traj_dir) / "cam0.mp4"
            if not cam0.exists():
                # Some trajs may have no cam0 (corrupt download etc.) — skip
                n_skipped += 1
                continue

            cfg = {
                "traj_id":       traj_id,
                "dataset":       args.dataset_name,
                "split":         split,
                "source_mp4":    str(cam0.resolve()),
                "n_frames":      args.n_frames,
                "fps":           args.fps,
                "height":        args.height,
                "width":         args.width,
            }

            out_dir = baseline_output_dir(
                args.output_root, "svd",
                args.dataset_name, split, traj_id,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "svd_config.json", "w") as f:
                json.dump(cfg, f, indent=2)

    print(f"\n=== SVD convert_data complete ===")
    print(f"  total entries:      {n_total}")
    print(f"  skipped (no cam0):  {n_skipped}")
    print(f"  configs written:    {n_total - n_skipped}")
    print(f"\nNext step:")
    print(f"  python -m eval.baseline.svd.run_eval "
          f"--output-root {args.output_root} --dataset-name {args.dataset_name} "
          f"--svd-ckpt $SVD_CKPT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
