"""Step 2: Standardize curated SSv2 clips.

For each clip in outputs/curated_clips.json:
  * read .webm
  * trim/pad to [min_frames, max_frames] frames
  * resize: shortest side -> image_size, then center-crop to (image_size, image_size)
  * write as mp4 at native fps (12 for SSv2)
  * write meta.json + cameras.json (default intrinsics, identity extrinsics)

Output:
  outputs/data/B_ssv2_<clip_id>_<verb>/
      cam0.mp4
      meta.json
      cameras.json
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


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def _resize_and_center_crop(frame: np.ndarray, size: int) -> np.ndarray:
    """Shortest-side resize to `size`, then center-crop to (size, size)."""
    from PIL import Image
    h, w = frame.shape[:2]
    if h < w:
        new_h, new_w = size, int(round(w * size / h))
    else:
        new_w, new_h = size, int(round(h * size / w))
    pil = Image.fromarray(frame).resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    pil = pil.crop((left, top, left + size, top + size))
    return np.asarray(pil)


def _read_all_frames(video_path: Path) -> list:
    """Read every frame from .webm or .mp4 as list of uint8 (H, W, 3)."""
    import imageio
    reader = imageio.get_reader(str(video_path))
    frames = []
    try:
        for f in reader:
            if f.ndim == 2:
                f = np.stack([f] * 3, axis=-1)
            if f.shape[-1] == 4:
                f = f[..., :3]
            frames.append(f)
    finally:
        reader.close()
    return frames


def _build_default_cameras(image_size: int, fovy_deg: float = 55.0) -> dict:
    """Single-camera config for Dataset-B: pinhole K from FOV; extrinsics = identity.

    Stored under camera key 'cam0' so the dataloader (cam-name agnostic) reads
    the same way as Dataset-A.
    """
    fovy = np.deg2rad(fovy_deg)
    fy = image_size / (2 * np.tan(fovy / 2))
    fx = fy
    return {
        "cam0": {
            "intrinsics": {
                "fx": float(fx), "fy": float(fy),
                "cx": float(image_size / 2), "cy": float(image_size / 2),
                "width": image_size, "height": image_size,
            },
            # Identity world->cam: world frame = camera frame for single-view
            "extrinsics": {
                "world_to_camera_4x4": np.eye(4, dtype=np.float64).tolist(),
            },
        }
    }


def _process_one(args_tuple):
    """Worker: standardize a single clip."""
    clip, out_root, image_size, min_frames, max_frames, pad_short, fps, fovy_deg = args_tuple

    traj_id = f"B_ssv2_{clip['clip_id']}_{clip['our_verb']}"
    traj_dir = Path(out_root) / traj_id
    out_mp4 = traj_dir / "cam0.mp4"
    out_meta = traj_dir / "meta.json"

    if out_mp4.exists() and out_meta.exists():
        return traj_id, "skipped", 0.0

    t0 = time.time()
    try:
        frames = _read_all_frames(Path(clip["video_path"]))
    except Exception as e:  # noqa: BLE001
        return traj_id, f"read-error: {type(e).__name__}: {e}", time.time() - t0

    n_total = len(frames)
    if n_total < min_frames:
        if not pad_short:
            return traj_id, f"too-short: {n_total} < {min_frames}", time.time() - t0
        frames = frames + [frames[-1]] * (min_frames - n_total)
    elif n_total > max_frames:
        # center-trim
        start = (n_total - max_frames) // 2
        frames = frames[start: start + max_frames]
    n_kept = len(frames)

    # resize+crop
    try:
        frames = [_resize_and_center_crop(f, image_size) for f in frames]
    except Exception as e:  # noqa: BLE001
        return traj_id, f"resize-error: {type(e).__name__}: {e}", time.time() - t0

    # write
    traj_dir.mkdir(parents=True, exist_ok=True)
    try:
        import imageio
        writer = imageio.get_writer(
            str(out_mp4),
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        try:
            for f in frames:
                writer.append_data(f)
        finally:
            writer.close()
    except Exception as e:  # noqa: BLE001
        return traj_id, f"write-error: {type(e).__name__}: {e}", time.time() - t0

    # cameras.json + meta.json
    cameras = _build_default_cameras(image_size, fovy_deg=fovy_deg)
    with open(traj_dir / "cameras.json", "w") as f:
        json.dump(cameras, f, indent=2)

    meta = {
        "traj_id":      traj_id,
        "obj_id":       clip["clip_id"],          # for compatibility with Dataset-A loader
        "obj_category": "ssv2_realworld",         # generic, since SSv2 has no fixed object class
        "task_name":    clip["our_verb"],
        "n_frames":     n_kept,
        "fps":          fps,
        "image_size":   image_size,
        "source":       "ssv2",
        "source_split": clip["source_split"],
        "raw_label":    clip["raw_label"],
        "template":     clip["template"],
        "placeholders": clip["placeholders"],
        "n_frames_original": n_total,
        # Mark for the dataloader / future code that this is a single-view real
        # video (not a 3-view simulation).
        "object_type":  "real_video",
        "n_cameras":    1,
        "is_composition": False,
        "eval_only":    False,
    }
    with open(traj_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return traj_id, "ok", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips",  default="outputs/curated_clips.json")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out",    default="outputs/data/")
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only first N clips (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(Path(args.config))
    clip_cfg = cfg["clip"]
    workers = args.num_workers or cfg["runtime"].get("num_workers", 8)

    with open(args.clips) as f:
        clips = json.load(f)["clips"]
    if args.limit:
        clips = clips[: args.limit]
    logging.info("Standardizing %d clips with %d workers", len(clips), workers)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    job_args = [
        (
            c,
            str(out_root),
            clip_cfg["image_size"],
            clip_cfg["min_frames"],
            clip_cfg["max_frames"],
            clip_cfg.get("pad_short_clips", True),
            clip_cfg["fps"],
            cfg["gs"]["default_fovy_deg"],
        )
        for c in clips
    ]

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
            if n_done % 25 == 0 or n_done == len(job_args):
                avg = sum(times[-100:]) / max(len(times[-100:]), 1)
                eta_s = avg * (len(job_args) - n_done)
                print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} "
                      f"fail={n_fail}  avg={avg:.1f}s/clip  eta={eta_s/60:.1f}min")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_process_one, ja): ja[0]["clip_id"] for ja in job_args}
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
                if n_done % 25 == 0 or n_done == len(job_args):
                    avg = sum(times[-100:]) / max(len(times[-100:]), 1)
                    elapsed = time.time() - t_start
                    eta_s = avg * (len(job_args) - n_done) / max(workers, 1)
                    print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} "
                          f"fail={n_fail}  avg={avg:.1f}s/clip  "
                          f"elapsed={elapsed/60:.1f}min  eta={eta_s/60:.1f}min")

    total = time.time() - t_start
    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail} in {total/60:.1f}min")


if __name__ == "__main__":
    main()
