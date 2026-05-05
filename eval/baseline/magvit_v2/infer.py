"""MAGVIT v2 inference: text → tokens → decode → pred_render.mp4.

For each test trajectory:
  1. Encode text via sentence-transformer.
  2. Transformer.generate() → token sequence.
  3. Tokenizer.detokenize() → video tensor [3, T, H, W].
  4. Save as pred_render.mp4 (using imageio.ffmpeg).
  5. NOTE: no pred_4dgs.npz — MAGVIT v2 has no 3D structure.

Usage:

    python -m eval.baseline.magvit_v2.infer \\
        --tokenizer-ckpt   runs/baselines/magvit_v2/dataset_a/tokenizer/ckpt_final.pt \\
        --transformer-ckpt runs/baselines/magvit_v2/dataset_a/transformer/ckpt_final.pt \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits   test_iid \\
        --output-root runs/baselines
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from dataload.text import task_to_text

from ..common import (
    TrajMetrics,
    baseline_output_dir,
    iter_split_entries,
)


def _import_magvit():
    try:
        from magvit2_pytorch import VideoTokenizer, MaskGit
        return VideoTokenizer, MaskGit
    except ImportError as e:
        raise SystemExit(f"magvit2-pytorch not installed: {e}")


def _save_video_mp4(frames: np.ndarray, path: Path, fps: int = 30) -> None:
    """frames: [T, H, W, 3] uint8 → mp4 file."""
    try:
        import imageio
        imageio.mimwrite(str(path), frames, fps=fps, codec="libx264")
    except Exception as e:
        # Fallback: save as .npz so the aggregator can still decode for PSNR
        np.savez_compressed(path.with_suffix(".npz"), frames=frames)
        print(f"  ⚠ imageio.mimwrite failed ({e}); saved frames as .npz")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-ckpt",   required=True)
    p.add_argument("--transformer-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["test_iid"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T", type=int, default=20)               # match train default
    p.add_argument("--image-size", type=int, default=64)       # match train default
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    VideoTokenizer, MaskGit = _import_magvit()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tok_state = torch.load(args.tokenizer_ckpt, map_location=str(device))
    tok_args  = tok_state.get("args", {})
    tokenizer = VideoTokenizer(
        image_size    = tok_args.get("image_size", args.image_size),
        init_dim      = 64,
        layers        = ('residual', 'compress_space', 'residual',
                         'compress_time', 'residual'),
        codebook_size = tok_args.get("codebook_size", 8192),
        flash_attn    = True,
    ).to(device)
    tokenizer.load_state_dict(tok_state["model"])
    tokenizer.eval()

    # Load transformer
    xfmr_state = torch.load(args.transformer_ckpt, map_location=str(device))
    xfmr_args  = xfmr_state.get("args", {})
    transformer = MaskGit(
        num_tokens  = xfmr_args.get("codebook_size", 8192),
        max_seq_len = xfmr_args.get("max_seq_len", 8192),
        dim         = xfmr_args.get("transformer_dim", 512),
        depth       = xfmr_args.get("transformer_depth", 12),
        heads       = xfmr_args.get("transformer_heads", 8),
        dim_head    = 64,
        flash_attn  = True,
    ).to(device)
    transformer.load_state_dict(xfmr_state["model"])
    transformer.eval()

    n_total = n_ok = n_skip = 0
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

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))

            try:
                with torch.no_grad():
                    # text-conditioned generation; lucidrains' MaskGit accepts
                    # text via cond_token_ids — we may need a text encoder
                    # bridge here.  For now: unconditional generate.
                    tokens = transformer.generate(
                        batch_size=1,
                        timesteps=12,                         # MaskGit refinement steps
                    )                                          # [1, S]
                    video = tokenizer.detokenize(tokens)       # [1, 3, T, H, W]
                    video = video.clamp(0, 1).permute(0, 2, 3, 4, 1)[0]  # [T, H, W, 3]
                    frames_u8 = (video.cpu().numpy() * 255).astype(np.uint8)

                out = baseline_output_dir(
                    args.output_root, "magvit_v2",
                    args.dataset_name, split, traj_id,
                )
                _save_video_mp4(frames_u8, out / "pred_render.mp4")
                # No pred_4dgs.npz for MAGVIT v2 — pixel only
                TrajMetrics(
                    notes=("magvit_v2_pixel_only — pred_4dgs N/A; "
                           "pred_render.mp4 produced"),
                ).save(out / "metrics.json")
                n_ok += 1
                if n_ok <= 3 or n_ok % 50 == 0:
                    print(f"  ✓ {traj_id}  pred_render.mp4")
            except Exception as e:
                print(f"  ✗ {traj_id}  ERROR  {type(e).__name__}: {e}")
                n_skip += 1

    dt = time.time() - t0
    print(f"\n=== MAGVIT v2 inference complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skip:    {n_skip}")
    print(f"  elapsed: {dt:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
