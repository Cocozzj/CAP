"""MAGVIT v2 training using lucidrains' magvit2-pytorch.

Two stages:
  Stage 1: VideoTokenizer (3D causal CNN VQ-VAE)
  Stage 2: TransformerDecoder (next-token prediction, text-conditioned)

This wrapper is a thin shell — the heavy lifting is in `magvit2_pytorch`.
Install with `pip install magvit2-pytorch`.

Usage:

    # Stage 1
    python -m eval.baseline.magvit_v2.train --stage tokenizer \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --epochs 50 \\
        --output-dir runs/baselines/magvit_v2/dataset_a/tokenizer

    # Stage 2
    python -m eval.baseline.magvit_v2.train --stage transformer \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --tokenizer-ckpt runs/baselines/magvit_v2/dataset_a/tokenizer/ckpt_final.pt \\
        --epochs 100 \\
        --output-dir runs/baselines/magvit_v2/dataset_a/transformer
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import MAGVITVideoDataset, collate_magvit


def _import_magvit():
    """Import lucidrains' magvit2-pytorch lazily — fails clearly if not installed."""
    try:
        from magvit2_pytorch import VideoTokenizer, MaskGit
        return VideoTokenizer, MaskGit
    except ImportError as e:
        raise SystemExit(
            "magvit2-pytorch not installed.  Run:\n"
            "  pip install magvit2-pytorch\n"
            f"Original error: {e}"
        )


def _setup_ddp():
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, local_rank
    return False, 0, 0


def _cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_tokenizer(args, is_ddp, rank, device, is_main) -> int:
    VideoTokenizer, _ = _import_magvit()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ds = MAGVITVideoDataset(args.manifest, args.data_dir, split="train",
                             T=args.T, image_size=args.image_size, cam_index=args.cam_index)
    sampler = DistributedSampler(ds, seed=args.seed) if is_ddp else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_magvit,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=2,
    )
    if is_main:
        print(f"MAGVIT tokenizer  samples={len(ds)}  cam={args.cam_index}  "
              f"image_size={args.image_size}")

    # Build the tokenizer (lucidrains' default config — adjust if needed)
    tokenizer = VideoTokenizer(
        image_size       = args.image_size,
        init_dim         = 64,
        layers           = ('residual', 'compress_space', 'residual',
                            'compress_time', 'residual'),
        codebook_size    = args.codebook_size,
        flash_attn       = True,
    ).to(device)

    if is_ddp:
        tokenizer = DDP(tokenizer, device_ids=[device.index],
                         find_unused_parameters=False)

    opt = torch.optim.AdamW(tokenizer.parameters(), lr=args.lr, betas=(0.9, 0.99))

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)        # [B, T, 3, H, W]
            # lucidrains expects [B, 3, T, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()
            opt.zero_grad(set_to_none=True)
            loss = tokenizer(video, return_loss=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
            global_step += 1
            if is_main and global_step % args.log_every == 0:
                print(f"[tok epoch={epoch} step={global_step}] loss={loss.item():.4f}")
        if is_main:
            print(f"  epoch {epoch + 1}/{args.epochs}  avg={total_loss/max(n,1):.4f}  "
                  f"dt={time.time()-t0:.1f}s")

    if is_main:
        tgt = tokenizer.module if hasattr(tokenizer, "module") else tokenizer
        torch.save({"model": tgt.state_dict(), "args": vars(args)},
                   out_dir / "ckpt_final.pt")
        print(f"  ✔ saved {out_dir/'ckpt_final.pt'}")
    return global_step


def train_transformer(args, is_ddp, rank, device, is_main) -> int:
    VideoTokenizer, MaskGit = _import_magvit()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen tokenizer
    tok_state = torch.load(args.tokenizer_ckpt, map_location=str(device))
    tok_args  = tok_state.get("args", {})
    tokenizer = VideoTokenizer(
        image_size    = tok_args.get("image_size", args.image_size),
        init_dim      = 64,
        layers        = ('residual', 'compress_space', 'residual',
                         'compress_time', 'residual'),
        codebook_size = tok_args.get("codebook_size", args.codebook_size),
        flash_attn    = True,
    ).to(device)
    tokenizer.load_state_dict(tok_state["model"])
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    # Build text-conditioned transformer (MaskGit is the maskgit one; if you
    # want pure causal AR, replace with MagViT2Transformer or similar)
    transformer = MaskGit(
        num_tokens       = args.codebook_size,
        max_seq_len      = args.max_seq_len,
        dim              = args.transformer_dim,
        depth            = args.transformer_depth,
        heads            = args.transformer_heads,
        dim_head         = 64,
        flash_attn       = True,
    ).to(device)

    if is_ddp:
        transformer = DDP(transformer, device_ids=[device.index],
                           find_unused_parameters=False)

    ds = MAGVITVideoDataset(args.manifest, args.data_dir, split="train",
                             T=args.T, image_size=args.image_size, cam_index=args.cam_index)
    sampler = DistributedSampler(ds, seed=args.seed) if is_ddp else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_magvit,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=2,
    )
    if is_main:
        print(f"MAGVIT transformer samples={len(ds)}  max_seq_len={args.max_seq_len}")

    opt = torch.optim.AdamW(transformer.parameters(), lr=args.lr, betas=(0.9, 0.99))

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time()
        total_loss = 0.0
        n = 0
        for batch in loader:
            video = batch["video"].to(device, non_blocking=True)
            video = video.permute(0, 2, 1, 3, 4).contiguous()           # [B, 3, T, H, W]
            with torch.no_grad():
                # tokens shape [B, S]  (S = compressed T × H/8 × W/8)
                tokens = tokenizer.tokenize(video)

            opt.zero_grad(set_to_none=True)
            loss = transformer(tokens, return_loss=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n += 1
            global_step += 1
            if is_main and global_step % args.log_every == 0:
                print(f"[xfmr epoch={epoch} step={global_step}] loss={loss.item():.4f}")
        if is_main:
            print(f"  epoch {epoch + 1}/{args.epochs}  avg={total_loss/max(n,1):.4f}  "
                  f"dt={time.time()-t0:.1f}s")

    if is_main:
        tgt = transformer.module if hasattr(transformer, "module") else transformer
        torch.save({"model": tgt.state_dict(), "args": vars(args)},
                   out_dir / "ckpt_final.pt")
        print(f"  ✔ saved {out_dir/'ckpt_final.pt'}")
    return global_step


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["tokenizer", "transformer"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    # Simplified defaults: smaller resolution + fewer frames + smaller model
    p.add_argument("--T",          type=int, default=20,        # ↓ from 30
                   help="frames per video (simplified)")
    p.add_argument("--image-size", type=int, default=64,        # ↓ from 128
                   help="64×64 reduces compute 4× vs 128×128")
    p.add_argument("--cam-index",  type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)         # ↑ from 4 (smaller frames fit)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--epochs",     type=int, default=None,
                   help="default: 20 for tokenizer, 50 for transformer")
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--log-every",  type=int, default=20)

    p.add_argument("--codebook-size",   type=int, default=2048)  # ↓ from 8192

    # Stage 2 specific — smaller transformer
    p.add_argument("--tokenizer-ckpt",   default=None)
    p.add_argument("--max-seq-len",       type=int, default=2048)  # ↓ from 8192
    p.add_argument("--transformer-dim",   type=int, default=256)   # ↓ from 512
    p.add_argument("--transformer-depth", type=int, default=6)     # ↓ from 12
    p.add_argument("--transformer-heads", type=int, default=8)

    args = p.parse_args(argv)

    # Apply per-stage epoch default if not provided
    if args.epochs is None:
        args.epochs = 20 if args.stage == "tokenizer" else 50

    is_ddp, rank, local_rank = _setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    if args.stage == "tokenizer":
        train_tokenizer(args, is_ddp, rank, device, is_main)
    else:
        if args.tokenizer_ckpt is None:
            raise SystemExit("--tokenizer-ckpt is required for stage=transformer")
        train_transformer(args, is_ddp, rank, device, is_main)

    _cleanup_ddp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
