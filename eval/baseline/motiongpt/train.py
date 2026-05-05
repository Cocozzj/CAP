"""Fine-tune T5 on (text → motion-token sequence) for the MotionGPT baseline.

Two-stage approach:
  1. Reuse the FlatVQVAE motion VQ-codebook (pre-trained on our pose deltas)
     as the motion vocabulary.
  2. Extend T5's tokenizer with K motion tokens + 2 special markers.
  3. Fine-tune T5 to predict (text → <motion_start> <m_x> ... <motion_end>).

T5 is loaded from HuggingFace (``t5-small`` by default — fits ~60M params).
For our small dataset (~1650 samples), this is intentional to avoid overfit.

Usage:

    python -m eval.baseline.motiongpt.train \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --vqvae-ckpt runs/baselines/flat_vqvae/dataset_a/vqvae/ckpt_final.pt \\
        --t5-name t5-small \\
        --output-dir runs/baselines/motiongpt/dataset_a/t5 \\
        --epochs 30
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

from ..flat_vqvae.vqvae import FlatVQVAE
from .data import (
    MGSpecialTokens,
    MotionGPTDataset,
    collate_mg,
    format_input_text,
    format_target_motion,
)


# ──────────────────────────────────────────────────────────────────────
# DDP setup
# ──────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────
# Tokenizer extension
# ──────────────────────────────────────────────────────────────────────

def extend_tokenizer(tokenizer, K: int, specials: MGSpecialTokens = MGSpecialTokens()):
    """Add motion-token vocabulary to a HuggingFace tokenizer."""
    new_tokens = specials.all_special_tokens(K)
    n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    return tokenizer, n_added


# ──────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────

def train_motiongpt(args, is_ddp, rank, device, is_main) -> int:
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load pre-trained VQ-VAE for tokenizing motion ──
    vq_state = torch.load(args.vqvae_ckpt, map_location=str(device))
    vq_args  = vq_state.get("args", {})
    vqvae = FlatVQVAE(
        in_dim=7,
        hidden=vq_args.get("hidden", 128),
        code_dim=vq_args.get("code_dim", 32),
        K=vq_args.get("K", 64),
    ).to(device)
    vqvae.load_state_dict(vq_state["model"])
    vqvae.eval()
    for p in vqvae.parameters():
        p.requires_grad_(False)
    K_motion = vqvae.quantizer.K

    # ── Load T5 + extended tokenizer ──
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        raise SystemExit(
            "transformers not installed — run: pip install transformers\n"
            "(needed for T5 backbone in MotionGPT baseline)"
        )

    tokenizer = T5Tokenizer.from_pretrained(args.t5_name)
    tokenizer, n_added = extend_tokenizer(tokenizer, K=K_motion)
    if is_main:
        print(f"⏬ Loaded {args.t5_name}; added {n_added} motion tokens "
              f"(K={K_motion} + 2 special)")

    model = T5ForConditionalGeneration.from_pretrained(args.t5_name).to(device)
    model.resize_token_embeddings(len(tokenizer))   # accommodate new tokens

    if is_ddp:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)

    # ── Dataset / loader ──
    ds = MotionGPTDataset(args.manifest, args.data_dir, split="train", T=args.T)
    sampler = DistributedSampler(ds, seed=args.seed) if is_ddp else None
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_mg,
        pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0), prefetch_factor=2,
    )
    if is_main:
        print(f"Train  samples={len(ds)}  steps/epoch={len(loader)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    global_step = 0
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        t0 = time.time(); total_loss = 0.0; n = 0
        for batch in loader:
            deltas = batch["deltas"].to(device, non_blocking=True)
            texts  = batch["texts"]

            # Encode pose deltas → motion token IDs (frozen VQ-VAE)
            with torch.no_grad():
                ids = vqvae.encode_to_ids(deltas)                     # [B, T-1] int

            # Build T5 source / target strings
            src_texts = [format_input_text(t) for t in texts]
            tgt_texts = [format_target_motion(ids[i].cpu().tolist()) for i in range(ids.shape[0])]

            src_enc = tokenizer(src_texts, padding=True, truncation=True,
                                 max_length=64, return_tensors="pt").to(device)
            tgt_enc = tokenizer(tgt_texts, padding=True, truncation=True,
                                 max_length=args.T + 8, return_tensors="pt").to(device)
            labels = tgt_enc.input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100   # ignore pad in loss

            opt.zero_grad(set_to_none=True)
            out = model(
                input_ids=src_enc.input_ids,
                attention_mask=src_enc.attention_mask,
                labels=labels,
            )
            loss = out.loss
            loss.backward()
            opt.step()

            total_loss += float(loss.item()); n += 1; global_step += 1
            if is_main and global_step % args.log_every == 0:
                print(f"[mg epoch={epoch} step={global_step}] loss={loss.item():.4f}")

        if is_main:
            print(f"  epoch {epoch + 1}/{args.epochs}  avg={total_loss/max(n,1):.4f}  "
                  f"dt={time.time()-t0:.1f}s")

    if is_main:
        target = model.module if hasattr(model, "module") else model
        out_dir.mkdir(parents=True, exist_ok=True)
        target.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        # Persist link to the VQ-VAE that defines our motion vocabulary
        with open(out_dir / "config_extra.json", "w") as f:
            import json as _json
            _json.dump({"vqvae_ckpt": str(Path(args.vqvae_ckpt).resolve()),
                          "K_motion":   int(K_motion),
                          "T":          args.T,
                          "args":       vars(args)}, f, indent=2)
        print(f"  ✔ saved {out_dir}/")
    return global_step


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--vqvae-ckpt", required=True,
                   help="path to FlatVQVAE ckpt (motion vocabulary source)")
    p.add_argument("--t5-name",    default="t5-small",
                   help="HuggingFace T5 variant: t5-small / t5-base / t5-large")
    p.add_argument("--T",          type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers",type=int, default=4)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--lr",         type=float, default=5e-5,
                   help="T5 fine-tune LR; smaller than VQ-VAE since pretrained")
    p.add_argument("--seed",       type=int, default=0)
    p.add_argument("--log-every",  type=int, default=20)
    args = p.parse_args(argv)

    is_ddp, rank, local_rank = _setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (rank == 0)

    train_motiongpt(args, is_ddp, rank, device, is_main)
    _cleanup_ddp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
