"""Step 3: Estimate monocular depth for each standardized clip.

Per the config (depth.estimate_first_frame_only), we only run the model on
the FIRST frame — this is the only frame we need for init_gs.ply. Adding
per-frame depth video is ~30x more compute and we don't use it during
training (Dataset-B has no multi-view supervision; depth is only a weak
prior at t=0).

Output per clip:
    outputs/data/<traj_id>/depth.npz
        depth      : (H, W) float32 meters
        is_metric  : bool, False — depth is only relative-scaled to ~1.5m median
        model      : 'depth_anything_v2'
        variant    : 'vitl'
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.depth_estimator import DepthAnythingV2


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _read_first_frame(mp4_path: Path) -> np.ndarray:
    import imageio
    reader = imageio.get_reader(str(mp4_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/data")
    ap.add_argument("--config",   default="configs/default.yaml")
    ap.add_argument("--device",   default=None,
                    help="Override device (default cfg.runtime.device)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N (smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if depth.npz exists")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(Path(args.config))
    depth_cfg = cfg["depth"]
    device = args.device or cfg["runtime"].get("device", "cuda")

    if depth_cfg["model"] != "depth_anything_v2":
        raise SystemExit(f"only depth_anything_v2 supported in this script; "
                         f"got {depth_cfg['model']!r}")

    estimator = DepthAnythingV2(
        variant=depth_cfg.get("variant", "vitl"),
        device=device,
        scene_scale_m=1.5,
    )

    data_dir = Path(args.data_dir)
    traj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if args.limit:
        traj_dirs = traj_dirs[: args.limit]
    logging.info("Estimating depth for %d trajectories on %s", len(traj_dirs), device)

    n_done = n_ok = n_skip = n_fail = 0
    t_start = time.time()
    times = []

    for tdir in traj_dirs:
        out_path = tdir / "depth.npz"
        mp4 = tdir / "cam0.mp4"
        if not mp4.exists():
            n_fail += 1
            print(f"  {tdir.name}: missing cam0.mp4")
            continue
        if out_path.exists() and not args.force:
            n_skip += 1
            n_done += 1
            continue

        t0 = time.time()
        try:
            frame = _read_first_frame(mp4)
            depth = estimator.predict(frame)
            np.savez_compressed(
                str(out_path),
                depth=depth.astype(np.float32),
                is_metric=False,
                model="depth_anything_v2",
                variant=depth_cfg.get("variant", "vitl"),
            )
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  {tdir.name}: error {type(e).__name__}: {e}")
        n_done += 1
        times.append(time.time() - t0)

        if n_done % 25 == 0 or n_done == len(traj_dirs):
            avg = sum(times[-100:]) / max(len(times[-100:]), 1)
            elapsed = time.time() - t_start
            eta_s = avg * (len(traj_dirs) - n_done)
            print(f"[{n_done}/{len(traj_dirs)}] ok={n_ok} skip={n_skip} "
                  f"fail={n_fail}  avg={avg:.2f}s/clip  "
                  f"elapsed={elapsed/60:.1f}min  eta={eta_s/60:.1f}min")

    total = time.time() - t_start
    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail} in {total/60:.1f}min")


if __name__ == "__main__":
    main()
