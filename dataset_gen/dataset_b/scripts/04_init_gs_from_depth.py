"""Step 4: Initialize a 3DGS PLY for each clip from the first-frame RGB
+ DepthAnything-v2 depth map. CPU-only; should take ~1 min for 1000 clips.

Reads:
    outputs/data/<traj_id>/cam0.mp4
    outputs/data/<traj_id>/depth.npz
    outputs/data/<traj_id>/cameras.json
Writes:
    outputs/data/<traj_id>/init_gs.ply
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gs_from_depth import back_project, write_init_gs_ply


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


def _process_one(args_tuple):
    traj_dir, gs_cfg, force = args_tuple
    traj_dir = Path(traj_dir)
    out_path = traj_dir / "init_gs.ply"
    if out_path.exists() and not force:
        return traj_dir.name, "skipped", 0.0

    t0 = time.time()
    try:
        depth_path = traj_dir / "depth.npz"
        if not depth_path.exists():
            return traj_dir.name, "missing depth.npz", time.time() - t0
        d = np.load(depth_path)
        depth = d["depth"]

        rgb = _read_first_frame(traj_dir / "cam0.mp4")
        if rgb.shape[:2] != depth.shape:
            # Defensive: if depth was somehow different size, resize it
            from PIL import Image
            depth = np.asarray(
                Image.fromarray(depth.astype(np.float32)).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.BILINEAR
                )
            )

        with open(traj_dir / "cameras.json") as f:
            cams = json.load(f)
        intr = cams["cam0"]["intrinsics"]
        K = np.array([
            [intr["fx"], 0,           intr["cx"]],
            [0,           intr["fy"], intr["cy"]],
            [0,           0,           1],
        ], dtype=np.float64)

        xyz, rgb01 = back_project(
            rgb, depth, K,
            depth_min=gs_cfg.get("depth_min_m", 0.1),
            depth_max=gs_cfg.get("depth_max_m", 10.0),
            background_quantile=gs_cfg.get("background_mask_quantile", 0.95),
            n_points=gs_cfg.get("init_n_points", 10000),
            seed=hash(traj_dir.name) & 0x7FFFFFFF,
        )
        write_init_gs_ply(out_path, xyz, rgb01)
        return traj_dir.name, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return traj_dir.name, f"error: {type(e).__name__}: {e}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="outputs/data")
    ap.add_argument("--config",   default="configs/default.yaml")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(Path(args.config))
    gs_cfg = cfg["gs"]
    workers = args.num_workers or cfg["runtime"].get("num_workers", 8)

    data_dir = Path(args.data_dir)
    traj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if args.limit:
        traj_dirs = traj_dirs[: args.limit]
    logging.info("Building init_gs.ply for %d trajectories with %d workers",
                 len(traj_dirs), workers)

    job_args = [(str(d), gs_cfg, args.force) for d in traj_dirs]

    n_done = n_ok = n_skip = n_fail = 0
    t_start = time.time()
    times = []

    if workers <= 1:
        for ja in job_args:
            tid, status, dt = _process_one(ja)
            n_done += 1
            times.append(dt)
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  {tid}: {status}")
            if n_done % 50 == 0 or n_done == len(job_args):
                avg = sum(times[-100:]) / max(len(times[-100:]), 1)
                print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} fail={n_fail}  "
                      f"avg={avg:.2f}s/clip")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process_one, ja) for ja in job_args]
            for fut in as_completed(futures):
                tid, status, dt = fut.result()
                n_done += 1
                times.append(dt)
                if status == "ok":
                    n_ok += 1
                elif status == "skipped":
                    n_skip += 1
                else:
                    n_fail += 1
                    print(f"  {tid}: {status}")
                if n_done % 50 == 0 or n_done == len(job_args):
                    avg = sum(times[-100:]) / max(len(times[-100:]), 1)
                    elapsed = time.time() - t_start
                    eta_s = avg * (len(job_args) - n_done) / max(workers, 1)
                    print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} "
                          f"fail={n_fail}  avg={avg:.2f}s/clip  "
                          f"elapsed={elapsed/60:.1f}min  eta={eta_s/60:.1f}min")

    total = time.time() - t_start
    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail} in {total/60:.1f}min")


if __name__ == "__main__":
    main()
