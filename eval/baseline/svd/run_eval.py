"""Run Stable Video Diffusion (SVD) on every prepared trajectory and
write the generated video to ``pred_render.mp4`` in the unified
baseline output layout.

Workflow:
  1. ``convert_data.py`` writes ``svd_config.json`` with each trajectory's
     ``source_mp4`` and target generation length.
  2. This script instantiates SVD-XT once per worker, then for each
     trajectory: load source first frame → run SVD → save mp4.
  3. ``render_metrics.py`` (with ``is_pixel_only=True`` for "svd") reads
     the mp4 directly and computes PSNR / LPIPS / SSIM against GT cam0.

Sharded execution (multi-GPU):

    for SHARD in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$SHARD nohup python -m eval.baseline.svd.run_eval \\
        --output-root runs/baselines --dataset-name dataset_a \\
        --splits test_iid \\
        --svd-ckpt ~/SVD_ckpts/svd-xt \\
        --shard-index $SHARD --num-shards 4 \\
        > /tmp/svd_${SHARD}.log 2>&1 &
    done

Per-traj: ~5-10s on A100 with fp16 + 15 inference steps.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..common import TrajMetrics


# ──────────────────────────────────────────────────────────────────────
# SVD pipeline (lazy-loaded)
# ──────────────────────────────────────────────────────────────────────

def _load_pipeline(svd_ckpt: str, dtype: str = "fp16"):
    """Instantiate diffusers' StableVideoDiffusionPipeline once per worker."""
    import torch
    from diffusers import StableVideoDiffusionPipeline

    torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        svd_ckpt,
        torch_dtype=torch_dtype,
        variant="fp16" if dtype == "fp16" else None,
    ).to("cuda")
    # Memory-efficient attention helps a lot on A100; safe to attempt
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    # CPU-offload would slow us down at scale — keep on GPU
    return pipe


# ──────────────────────────────────────────────────────────────────────
# First-frame extraction from cam0.mp4
# ──────────────────────────────────────────────────────────────────────

def _read_first_frame(mp4_path: Path, target_w: int, target_h: int):
    """Return PIL.Image (RGB, target_w x target_h) of mp4's first frame."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(mp4_path))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return Image.fromarray(rgb)


# ──────────────────────────────────────────────────────────────────────
# Save SVD output → mp4
# ──────────────────────────────────────────────────────────────────────

def _save_frames_as_mp4(frames, mp4_path: Path, fps: int) -> None:
    """frames: list of PIL.Image (RGB).  Encode to H.264 mp4."""
    import imageio.v2 as imageio
    arr = [np.asarray(f.convert("RGB")) for f in frames]   # list of (H,W,3) uint8
    writer = imageio.get_writer(
        str(mp4_path), fps=fps, codec="libx264",
        macro_block_size=1,             # tolerate non-16-multiple H/W
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for img in arr:
            writer.append_data(img)
    finally:
        writer.close()


# ──────────────────────────────────────────────────────────────────────
# Single-traj inference
# ──────────────────────────────────────────────────────────────────────

def _run_one(
    pipe,
    cfg_path: Path,
    out_mp4:  Path,
    n_inference_steps: int,
    motion_bucket_id:  int,
    noise_aug_strength: float,
) -> tuple[bool, str]:
    """Run SVD on one trajectory's first frame; write out_mp4."""
    cfg = json.loads(cfg_path.read_text())
    src = Path(cfg["source_mp4"])
    if not src.exists():
        return False, f"source_mp4 missing: {src}"

    img = _read_first_frame(src, cfg["width"], cfg["height"])
    if img is None:
        return False, f"failed to read first frame of {src}"

    try:
        result = pipe(
            img,
            num_frames=cfg["n_frames"],
            num_inference_steps=n_inference_steps,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=noise_aug_strength,
            decode_chunk_size=8,         # A100 handles 8 fine; 4 if OOM
        )
    except Exception as e:
        return False, f"svd_inference: {type(e).__name__}: {e}"

    frames = result.frames[0]  # list[PIL.Image] of length n_frames
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        _save_frames_as_mp4(frames, out_mp4, fps=cfg["fps"])
    except Exception as e:
        return False, f"mp4_encode: {type(e).__name__}: {e}"

    return True, "ok"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--splits",       nargs="+", default=None)
    p.add_argument("--svd-ckpt", required=True,
                   help="path to local Stable Video Diffusion checkpoint dir "
                        "(e.g. ~/SVD_ckpts/svd-xt downloaded via "
                        "huggingface-cli download stabilityai/"
                        "stable-video-diffusion-img2vid-xt --local-dir ...)")
    # SVD inference params — defaults from diffusers' SVD-XT recipe
    p.add_argument("--steps", type=int, default=15,
                   help="num_inference_steps; 15 is recipe default, faster "
                        "than 25 with marginal quality loss for our use")
    p.add_argument("--motion-bucket", type=int, default=127,
                   help="motion_bucket_id (0-255); higher → larger motion. "
                        "127 = recipe default")
    p.add_argument("--noise-aug", type=float, default=0.02,
                   help="noise_aug_strength; 0.02 = recipe default")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip trajs that already have pred_render.mp4")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards",  type=int, default=1)
    args = p.parse_args(argv)
    assert 0 <= args.shard_index < args.num_shards

    base = Path(args.output_root) / "svd" / args.dataset_name
    if not base.exists():
        print(f"✗ no convert_data outputs at {base}", file=sys.stderr)
        print(f"  run convert_data.py first", file=sys.stderr)
        return 1

    splits = args.splits or [d.name for d in sorted(base.iterdir()) if d.is_dir()]

    # Collect (traj_dir, cfg_path, out_mp4) across all selected splits, then
    # apply round-robin sharding.  This keeps load balanced even if one
    # split is much larger than others.
    work: List[tuple[Path, Path, Path]] = []
    for split in splits:
        sd = base / split
        if not sd.exists():
            continue
        for traj_dir in sorted(sd.iterdir()):
            if not traj_dir.is_dir():
                continue
            cfg = traj_dir / "svd_config.json"
            mp4 = traj_dir / "pred_render.mp4"
            if not cfg.exists():
                continue
            if args.skip_existing and mp4.exists():
                continue
            work.append((traj_dir, cfg, mp4))

    work = [w for j, w in enumerate(work) if (j % args.num_shards) == args.shard_index]
    if args.limit is not None:
        work = work[: args.limit]
    if args.num_shards > 1:
        print(f"[shard {args.shard_index}/{args.num_shards}] {len(work)} trajs to process")

    if not work:
        print(f"nothing to do for shard {args.shard_index}/{args.num_shards}")
        return 0

    print(f"⏬ loading SVD from {args.svd_ckpt} …")
    t_load = time.time()
    pipe = _load_pipeline(args.svd_ckpt, dtype=args.dtype)
    print(f"  loaded in {time.time()-t_load:.1f}s")

    n_ok = n_fail = 0
    t0 = time.time()
    for i, (traj_dir, cfg, out_mp4) in enumerate(work):
        success, msg = _run_one(
            pipe, cfg, out_mp4,
            n_inference_steps=args.steps,
            motion_bucket_id=args.motion_bucket,
            noise_aug_strength=args.noise_aug,
        )
        if not success:
            TrajMetrics(notes=f"svd_failed: {msg[:200]}").save(traj_dir / "metrics.json")
            n_fail += 1
            if n_fail <= 3:
                print(f"  ✗ {traj_dir.name}  {msg}")
            continue
        # Mark as ready for downstream metric computation.
        TrajMetrics(notes="svd_pending_eval").save(traj_dir / "metrics.json")
        n_ok += 1
        if n_ok <= 3 or n_ok % 50 == 0:
            elapsed = time.time() - t0
            avg = elapsed / max(i + 1, 1)
            eta = avg * (len(work) - (i + 1))
            print(f"  ✓ {traj_dir.name}  ({n_ok}/{len(work)}, "
                  f"avg {avg:.1f}s/traj, ETA {eta/60:.1f}min)")

    print(f"\n=== SVD shard {args.shard_index}/{args.num_shards} complete ===")
    print(f"  total:   {len(work)}")
    print(f"  ok:      {n_ok}")
    print(f"  failed:  {n_fail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
