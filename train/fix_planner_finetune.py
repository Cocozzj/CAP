"""Recover from planner mode collapse: re-initialise the task codebook
and lang.proj_head, freeze encoder/executor, and fine-tune planner with
strengthened InfoNCE (post-VQ + pre-VQ).

Why this exists
---------------
Our main_a/seed_*/ckpt/main_exp_final.pt all exhibit catastrophic VQ
codebook collapse:

    pairwise cos sim of 128 task_codebook entries: min 0.995, max 1.000

Since every entry points in the same direction, every text query maps
to the same task_id at inference (any nearest-neighbor search returns a
near-uniform argmin).  Mode collapse cascades: planner emits identical
plan tokens regardless of input text.

This script:
  1. Loads the pretrained main_exp_final.pt.
  2. Re-initialises:
       • lang.proj_head           — fresh weights (Xavier)
       • task_tok.quantizer.codebook — k-means centroids on text_emb
                                       sampled from the train set
  3. Freezes encoder + executor.  Trainable: lang.proj_head, task_tok,
     and planner.cvae (so the AR decoder can re-adapt to the new
     codebook).
  4. Trains for ``--epochs`` (default 10) using the patched loss
     (lambda_nce ↑ from 0.1 to 1.0, lambda_nce_preVQ added at 1.0,
     lambda_vq_task ↓ from 1.0 to 0.3, lambda_entropy ↑ to 0.30).
  5. Saves to ``runs/fix_planner_a/seed_<X>/ckpt/main_exp_final.pt`` so
     the existing eval pipeline picks it up directly.

Usage:
    python -m train.fix_planner_finetune \\
        --pretrained-ckpt runs/main_a/seed_0/ckpt/main_exp_final.pt \\
        --output-dir       runs/fix_planner_a/seed_0 \\
        --config           configs/config.yaml \\
        --loss-config      configs/loss.yaml \\
        --epochs           10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# Heavy project imports — do at top so import errors surface early.
from dataload.text  import task_to_text


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def kmeans_init_codebook(
    samples:    torch.Tensor,    # [N, D]
    n_clusters: int,
    n_iters:    int = 25,
    seed:       int = 0,
) -> torch.Tensor:
    """Pure-PyTorch k-means.  Returns [n_clusters, D] centroids."""
    g = torch.Generator(device=samples.device).manual_seed(seed)
    n, d = samples.shape
    if n < n_clusters:
        # Not enough samples — pad by repeating
        pad = torch.randint(0, n, (n_clusters - n,), generator=g, device=samples.device)
        idx = torch.cat([torch.arange(n, device=samples.device), pad])
    else:
        idx = torch.randperm(n, generator=g, device=samples.device)[:n_clusters]
    centroids = samples[idx].clone()

    for _ in range(n_iters):
        # Assign each sample to nearest centroid
        d2 = (samples.pow(2).sum(1, keepdim=True)
              - 2 * samples @ centroids.T
              + centroids.pow(2).sum(1))
        assign = d2.argmin(dim=1)
        # Recompute centroids
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(n_clusters, device=samples.device)
        new_centroids.index_add_(0, assign, samples)
        counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.float))
        nonempty = counts > 0
        new_centroids[nonempty] /= counts[nonempty].unsqueeze(1)
        # Reseed empty clusters from random samples
        empty = (~nonempty).nonzero(as_tuple=True)[0]
        if empty.numel() > 0:
            ridx = torch.randint(0, n, (empty.numel(),), generator=g, device=samples.device)
            new_centroids[empty] = samples[ridx]
        centroids = new_centroids
    return centroids


def reinit_proj_head(model) -> None:
    """Xavier-init the lang.proj_head (768 → task_dim) so it can re-learn
    a non-collapsed projection."""
    for m in model.planner.lang.proj_head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)


def reinit_task_codebook(model, train_text_emb: torch.Tensor) -> None:
    """Replace the collapsed task_codebook with k-means centroids of the
    given text_emb pool."""
    cb = model.planner.task_tok.quantizer.codebook   # nn.Embedding
    J, D = cb.weight.shape
    print(f"  re-init task_codebook (J={J}, D={D}) via k-means on "
          f"{train_text_emb.shape[0]} text_emb samples …")
    centroids = kmeans_init_codebook(train_text_emb.float(), J, n_iters=25)
    cb.weight.data.copy_(centroids)
    # Reset EMA buffers if present
    qz = model.planner.task_tok.quantizer
    if hasattr(qz, "ema_weight"):
        qz.ema_weight.copy_(centroids)
    if hasattr(qz, "ema_count"):
        qz.ema_count.fill_(1.0)
    if hasattr(qz, "usage_count"):
        qz.usage_count.zero_()
    if hasattr(qz, "step_count"):
        qz.step_count.zero_()


def freeze_module(m: torch.nn.Module) -> int:
    """Set all parameters in m to requires_grad=False; return param count."""
    n = 0
    for p in m.parameters():
        p.requires_grad_(False)
        n += p.numel()
    return n


def collect_train_text_embs(
    model,
    manifest_path: str,
    data_dir:      str,
    device:        torch.device,
    n_max:         int = 4000,
) -> torch.Tensor:
    """Sample up to ``n_max`` train trajectories' texts, run them through
    text_enc to collect text_emb (pre-projection!)."""
    from eval.baseline.common import iter_split_entries
    texts: List[str] = []
    for tid, _td, ent in iter_split_entries(manifest_path, data_dir, "train"):
        if isinstance(ent, dict):
            t = task_to_text(ent.get("task_name", ""), ent.get("obj_category", ""))
            if t:
                texts.append(t)
    random.shuffle(texts)
    texts = texts[:n_max]
    print(f"  collecting text_emb from {len(texts)} train texts …")
    embs: List[torch.Tensor] = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            chunk = texts[i:i + bs]
            v = model.planner.lang.encode(chunk)        # [B, task_dim]
            embs.append(v.detach().cpu())
    return torch.cat(embs, dim=0).to(device)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-ckpt", required=True,
                   help="path to main_exp_final.pt with collapsed codebook")
    p.add_argument("--output-dir",      required=True)
    p.add_argument("--config",          default="configs/config.yaml")
    p.add_argument("--loss-config",     default="configs/loss.yaml")
    p.add_argument("--manifest", default="dataset/dataset_a/manifest.json")
    p.add_argument("--data-dir", default="dataset/dataset_a/data")
    p.add_argument("--epochs",   type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr",       type=float, default=2e-4)
    p.add_argument("--seed",     type=int, default=0)
    args = p.parse_args(argv)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    out_dir = Path(args.output_dir)
    (out_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load pretrained model (with our sentence-transformers remap)
    from eval.baseline.ours.runner import load_model
    print(f"⏬ loading {args.pretrained_ckpt}")
    model = load_model(args.pretrained_ckpt, args.config, device)
    model.train()

    # 2. Collect training text embeddings (post current proj_head, just for k-means seed)
    #    We'll re-init proj_head right after, so this is just a starting
    #    seed — k-means doesn't need accurate embeddings, just diverse ones.
    pre_embs = collect_train_text_embs(
        model, args.manifest, args.data_dir, device, n_max=4000,
    )

    # 3. Re-init proj_head + task_codebook
    print("\n⤺ re-initialising collapsed components")
    reinit_proj_head(model)
    # After re-init, recompute embs (now from fresh proj_head)
    fresh_embs = collect_train_text_embs(
        model, args.manifest, args.data_dir, device, n_max=4000,
    )
    reinit_task_codebook(model, fresh_embs)

    # 4. Freeze encoder + executor + text_enc; only planner.lang.proj_head,
    #    task_tok, and planner.cvae stay trainable
    n_frozen = 0
    n_frozen += freeze_module(model.encoder)
    n_frozen += freeze_module(model.executor)
    n_frozen += freeze_module(model.planner.lang.text_enc)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n⏬ frozen {n_frozen / 1e6:.1f}M params; trainable {trainable / 1e6:.1f}M")

    # 5. Set up dataset (re-using the project's existing dataloader)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(args.loss_config) as f:
        loss_cfg = yaml.safe_load(f)["loss"]

    # Project uses DatasetA / DatasetB classes + collate_batch (see
    # train/trainer.py::build_dataset / _build_model_and_data).
    from dataload import DatasetA, collate_batch
    sh_dim = int(cfg["gs_param"]["gs_dimension"]) - 11
    T_train  = int(cfg.get("data", {}).get("T", 30))
    image_sz = int(cfg.get("data", {}).get("image_size", 256))
    train_set = DatasetA(
        manifest_path = args.manifest,
        data_dir      = args.data_dir,
        split         = "train",
        T             = T_train,
        image_size    = image_sz,
        c_sh          = sh_dim,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_batch,
        pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    print(f"\n⏬ train_loader: {len(train_loader)} steps/epoch  ({len(train_set)} samples)")

    # 6. Optimizer (only trainable params)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01,
    )

    # 7. Loss module — match trainer.py call signature
    from model.loss import CAPLoss
    from train.stages import StageSpec
    cap_loss = CAPLoss(cfg=loss_cfg)
    # Pretend we're in the FULL stage so all loss flags are on.  Render is
    # disabled (render_n_timesteps=0) because we only fine-tune planner;
    # rendering wastes compute and we don't need the visual loss for fix.
    spec = StageSpec(
        name="FIX_PLANNER", epochs=args.epochs, lr=args.lr,
        encoder=False, planner=True, executor=False,
        enable_physics=False, run_planner=True,
        render_n_timesteps=0,        # skip rendering — fastest path
    )
    spec.loss.enable_hier = True

    # 8. Training loop — mirrors trainer.train_epoch
    print(f"\n⏬ training for {args.epochs} epochs on {device}")
    total_steps = args.epochs * len(train_loader)
    global_step = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        sums = {"L_NCE": 0., "L_NCE_preVQ": 0., "planner_ce": 0.,
                "L_VQ_task": 0., "total": 0.}
        n_steps = 0
        for batch in train_loader:
            global_step += 1
            n_steps += 1

            frames    = batch["frames"].to(device, non_blocking=True)
            gs_params = [g.to(device=device) for g in batch["gs_params"]]
            condition = {
                "texts":              batch.get("text"),
                "sample_prob":        0.0,
                "render_n_timesteps": 0,
            }
            cameras = None
            if "intrinsics" in batch and "extrinsics" in batch:
                cameras = {
                    "intrinsics": batch["intrinsics"].to(device, non_blocking=True),
                    "extrinsics": batch["extrinsics"].to(device, non_blocking=True),
                }

            opt.zero_grad(set_to_none=True)
            try:
                training_out = model(
                    frames, gs_params=gs_params,
                    enable_physics=False, run_planner=True,
                    tau=0.1, condition=condition, cameras=cameras,
                )
                losses = cap_loss(
                    model=model, training_out=training_out,
                    gt={"frames": frames, "depth": None,
                        "text":   condition["texts"]},
                    spec=spec.loss,
                    step=global_step, total_steps=total_steps,
                )
            except Exception as e:
                if global_step <= 3:
                    import traceback
                    print(f"  ⚠ step {global_step} failed: {type(e).__name__}: {e}")
                    traceback.print_exc()
                continue

            total = losses["total"]
            if torch.isnan(total).any() or torch.isinf(total).any():
                if global_step <= 3:
                    print(f"  ⚠ step {global_step} total NaN/Inf — skip")
                continue
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0,
            )
            opt.step()

            for k in sums:
                v = losses.get(k)
                if v is not None and torch.is_tensor(v):
                    sums[k] += float(v.item())

            if global_step % 50 == 0:
                avg = {k: sums[k] / max(n_steps, 1) for k in sums}
                print(f"  step {global_step}  total={avg['total']:.4f}  "
                      f"NCE={avg['L_NCE']:.4f}  NCE_pre={avg['L_NCE_preVQ']:.4f}  "
                      f"VQ_task={avg['L_VQ_task']:.4f}  CE={avg['planner_ce']:.4f}")

        avg = {k: sums[k] / max(n_steps, 1) for k in sums}
        # Quick codebook diagnostic each epoch
        with torch.no_grad():
            cb = model.planner.task_tok.quantizer.codebook.weight
            ncb = F.normalize(cb, dim=-1)
            sim = ncb @ ncb.T
            mask = ~torch.eye(cb.shape[0], dtype=torch.bool, device=cb.device)
            cb_sim_mean = sim[mask].mean().item()
        print(f"epoch {epoch + 1}/{args.epochs}  "
              f"total={avg['total']:.4f}  NCE={avg['L_NCE']:.4f}  "
              f"NCE_pre={avg['L_NCE_preVQ']:.4f}  "
              f"VQ_task={avg['L_VQ_task']:.4f}  "
              f"codebook_cos={cb_sim_mean:.4f}  "
              f"dt={time.time()-t0:.1f}s")

        # Save checkpoint each epoch (overwrite final on each)
        torch.save(
            {"model": model.state_dict(),
             "epoch": epoch + 1,
             "global_step": global_step,
             "args": vars(args)},
            out_dir / "ckpt" / "main_exp_final.pt",
        )

    print(f"\n✔ saved to {out_dir / 'ckpt' / 'main_exp_final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
