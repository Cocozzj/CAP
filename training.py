"""
training.py — 3-stage curriculum trainer for CAP main experiment.

Stages (PDF f07d2c0a §2.4 / fdfa011c §5.1)
──────────────────────────────────────────
  Stage 0  RIGID    — 50 epochs, lr=1e-4   Encoder + Executor.rigid trainable;
                                            Planner frozen, physics OFF
  Stage 1  PHYSICS  — 20 epochs, lr=5e-5   Executor.deform only; rest frozen,
                                            physics ON
  Stage 2  FULL     — 30 epochs, lr=1e-5   All trainable, all losses on,
                                            CVAE β annealed 0.01 → 1.0,
                                            comm weight ramped 0.01 → 0.1

Features
────────
  - Multi-GPU via torch.distributed (DDP)
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping (PDF §5.1 explicitly required)
  - TensorBoard logging
  - Checkpoint policy: per-stage end + every N epochs, prune to last K
  - Resume from any checkpoint

Single-GPU smoke-test (random toy data)::

    python training.py --dataset toy --n-toy-samples 64

Single-GPU on real Dataset-A::

    python training.py --dataset dataset_a \\
        --manifest dataset_gen/dataset_a/outputs/manifest.json \\
        --data-dir dataset_gen/dataset_a/outputs/data \\
        --split train --T 30 --n-gs-points 10000 --image-size 256

Multi-GPU launch (8 GPUs, real data)::

    torchrun --nproc_per_node=8 training.py \\
        --dataset dataset_a \\
        --manifest dataset_gen/dataset_a/outputs/manifest.json \\
        --data-dir dataset_gen/dataset_a/outputs/data \\
        --batch-size 8 --num-workers 4

Resume::

    python training.py --resume runs/exp1/ckpt/stage1_done.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

from model import CAPModel, CAPLoss, TrainingStage
from dataloader import DatasetA, ToyDataset, collate_batch


# ══════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════

def set_global_seed(seed: int, *, rank: int = 0, deterministic: bool = False) -> None:
    """Seed every RNG we touch.  Per-rank offset keeps DDP workers diverse
    while still being reproducible from the same ``--seed``.

    Set ``deterministic=True`` to also force cuDNN deterministic kernels —
    slower but bit-exact across runs.  We leave it OFF by default because
    the 5-seed sweep (Tab 2) already absorbs run-to-run noise.
    """
    effective = seed + rank * 10_000
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.benchmark     = True


def _seed_dataloader_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn — derive each worker's seed from torch's."""
    base = torch.initial_seed() % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# ══════════════════════════════════════════════════════════════════════
# Stage schedule
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StageSpec:
    name:    str
    code:    int                   # TrainingStage.RIGID/PHYSICS/FULL
    epochs:  int
    lr:      float

DEFAULT_STAGES: List[StageSpec] = [
    StageSpec("RIGID",   TrainingStage.RIGID,   epochs=50, lr=1e-4),
    StageSpec("PHYSICS", TrainingStage.PHYSICS, epochs=20, lr=5e-5),
    StageSpec("FULL",    TrainingStage.FULL,    epochs=30, lr=1e-5),
]

# Local pipeline smoke-test: 1 epoch per stage, same lr.
SMOKE_STAGES: List[StageSpec] = [
    StageSpec("RIGID",   TrainingStage.RIGID,   epochs=1, lr=1e-4),
    StageSpec("PHYSICS", TrainingStage.PHYSICS, epochs=1, lr=5e-5),
    StageSpec("FULL",    TrainingStage.FULL,    epochs=1, lr=1e-5),
]


# ══════════════════════════════════════════════════════════════════════
# DDP utilities
# ══════════════════════════════════════════════════════════════════════

def setup_ddp() -> tuple[bool, int, int, int]:
    """Initialise distributed training if launched via torchrun.

    Returns: (is_ddp, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main(rank: int) -> bool:
    return rank == 0


# ══════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    stage: int,
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint atomically (write to .tmp then rename)."""
    state = {
        "model":       (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "stage":       stage,
        "epoch":       epoch,
        "global_step": global_step,
        "extra":       extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.rename(path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    state = torch.load(path, map_location=map_location)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    return state


def prune_checkpoints(ckpt_dir: Path, keep_last: int = 3) -> None:
    """Keep only the most recent ``keep_last`` mid-epoch checkpoints."""
    epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"),
                         key=lambda p: int(p.stem.split("_")[1]))
    for old in epoch_ckpts[:-keep_last]:
        try: old.unlink()
        except OSError: pass


# ══════════════════════════════════════════════════════════════════════
# Optimiser builder
# ══════════════════════════════════════════════════════════════════════

def build_optimizer(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Build Adam over only the trainable parameters at this stage.

    set_stage() flips requires_grad on/off, so we filter here per-call.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# ══════════════════════════════════════════════════════════════════════
# One epoch
# ══════════════════════════════════════════════════════════════════════

def train_epoch(
    *,
    model:     nn.Module,
    loss_fn:   CAPLoss,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    Optional[GradScaler],
    device:    torch.device,
    stage:     int,
    epoch:     int,
    global_step: int,
    total_steps: int,
    grad_clip: float = 1.0,
    log_every: int = 10,
    writer:    Optional[SummaryWriter] = None,
    is_main_proc: bool = True,
) -> int:
    """Run one training epoch, return updated global_step."""
    model.train()
    for batch_idx, batch in enumerate(loader):
        frames    = batch["frames"].to(device, non_blocking=True)
        gs_params = [g.to(device=device) for g in batch["gs_params"]]
        condition = {"texts": batch.get("text")}

        optimizer.zero_grad(set_to_none=True)

        # ── Forward (AMP if scaler is provided) ──
        if scaler is not None:
            with autocast():
                training_out = model(
                    frames, gs_params=gs_params, tau=1.0, condition=condition,
                )
                losses = loss_fn(
                    model=(model.module if hasattr(model, "module") else model),
                    training_out=training_out,
                    gt={"frames": None, "depth": None, "text": condition["texts"]},
                    stage=stage, step=global_step, total_steps=total_steps,
                )
                total = losses["total"]
        else:
            training_out = model(
                frames, gs_params=gs_params, tau=1.0, condition=condition,
            )
            losses = loss_fn(
                model=(model.module if hasattr(model, "module") else model),
                training_out=training_out,
                gt={"frames": None, "depth": None, "text": condition["texts"]},
                stage=stage, step=global_step, total_steps=total_steps,
            )
            total = losses["total"]

        # ── Backward + step (AMP-aware) ──
        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=grad_clip,
            )
            optimizer.step()

        global_step += 1

        # ── Logging ──
        if is_main_proc and (batch_idx % log_every == 0):
            msg = (f"[stage={stage} epoch={epoch} step={global_step}] "
                   f"total={total.item():.4f}")
            print(msg, flush=True)
            if writer is not None:
                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        writer.add_scalar(f"loss/{k}", v.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

    return global_step


# ══════════════════════════════════════════════════════════════════════
# Stage runner
# ══════════════════════════════════════════════════════════════════════

def run_stage(
    *,
    spec:        StageSpec,
    model:       nn.Module,
    loss_fn:     CAPLoss,
    loader:      DataLoader,
    sampler:     Optional[DistributedSampler],
    device:      torch.device,
    ckpt_dir:    Path,
    log_dir:     Path,
    use_amp:     bool,
    grad_clip:   float,
    save_every:  int,
    keep_last:   int,
    is_main_proc: bool,
    starting_global_step: int,
    is_ddp:      bool,
) -> int:
    """Train one stage, return the cumulative global_step at end."""
    base_model = model.module if hasattr(model, "module") else model
    base_model.set_stage(spec.code)

    # Re-init optimiser over current trainable params (set_stage may have flipped them)
    optimizer = build_optimizer(model, spec.lr)
    scaler    = GradScaler() if (use_amp and torch.cuda.is_available()) else None
    writer    = SummaryWriter(log_dir / spec.name) if is_main_proc else None

    total_iters = spec.epochs * len(loader)
    global_step = starting_global_step

    if is_main_proc:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== Stage {spec.name} ===")
        print(f"  epochs={spec.epochs}, lr={spec.lr}, trainable params={n_train:,}")

    for epoch in range(spec.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()
        global_step = train_epoch(
            model=model, loss_fn=loss_fn, loader=loader,
            optimizer=optimizer, scaler=scaler,
            device=device, stage=spec.code, epoch=epoch,
            global_step=global_step, total_steps=total_iters,
            grad_clip=grad_clip,
            writer=writer, is_main_proc=is_main_proc,
        )
        if is_main_proc:
            dt = time.time() - t0
            print(f"  epoch {epoch + 1}/{spec.epochs} done in {dt:.1f}s  "
                  f"step={global_step}", flush=True)

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                p = ckpt_dir / f"epoch_{global_step:08d}.pt"
                save_checkpoint(p, model, optimizer, scaler,
                                stage=spec.code, epoch=epoch + 1,
                                global_step=global_step,
                                extra={"stage_name": spec.name})
                prune_checkpoints(ckpt_dir, keep_last=keep_last)
                print(f"  ✔ saved {p.name}")

    # End-of-stage checkpoint
    if is_main_proc:
        p = ckpt_dir / f"stage_{spec.name.lower()}_done.pt"
        save_checkpoint(p, model, optimizer, scaler,
                        stage=spec.code, epoch=spec.epochs,
                        global_step=global_step,
                        extra={"stage_name": spec.name, "stage_done": True})
        print(f"  ★ stage-end ckpt: {p.name}")
        if writer is not None:
            writer.close()

    if is_ddp:
        dist.barrier()

    return global_step


# ══════════════════════════════════════════════════════════════════════
# Dataset factory
# ══════════════════════════════════════════════════════════════════════

def build_dataset(args, sh_dim: int):
    """Build the train Dataset based on ``--dataset``.

    Both backends produce the same per-sample dict shape, so the rest of
    the training loop is dataset-agnostic.
    """
    if args.dataset == "toy":
        return ToyDataset(n_samples=args.n_toy_samples, sh_dim=sh_dim)
    if args.dataset == "dataset_a":
        if not args.manifest or not args.data_dir:
            raise SystemExit("--dataset dataset_a requires --manifest and --data-dir")
        return DatasetA(
            manifest_path = args.manifest,
            data_dir      = args.data_dir,
            split         = args.split,
            T             = args.T,
            image_size    = args.image_size,
            c_sh          = sh_dim,
        )
    raise SystemExit(f"Unknown --dataset {args.dataset!r}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--loss-config", type=str, default="configs/loss.yaml")
    parser.add_argument("--out-dir",     type=str, default="results/main_exp")
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp",      action="store_true")
    parser.add_argument("--grad-clip",   type=float, default=1.0)
    parser.add_argument("--save-every",  type=int, default=5,
                        help="Save mid-stage checkpoint every N epochs")
    parser.add_argument("--keep-last",   type=int, default=3,
                        help="Keep only last K mid-stage checkpoints")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")

    # ── Dataset selection ──────────────────────────────────────────────
    parser.add_argument("--dataset",       type=str, default="toy",
                        choices=["toy", "dataset_a"],
                        help="Which dataset backend to use.")
    parser.add_argument("--n-toy-samples", type=int, default=64,
                        help="ToyDataset size (--dataset toy only)")
    parser.add_argument("--manifest",      type=str, default=None,
                        help="Path to outputs/manifest.json (--dataset dataset_a)")
    parser.add_argument("--data-dir",      type=str, default=None,
                        help="Path to outputs/data (--dataset dataset_a)")
    parser.add_argument("--split",         type=str, default="train")
    parser.add_argument("--T",             type=int, default=30,
                        help="frames per sample (>=10, %%5==0)")
    parser.add_argument("--image-size",    type=int, default=256)

    parser.add_argument("--seed", type=int, default=0,
                        help="Master seed (per-rank offset added inside).")
    parser.add_argument("--deterministic", action="store_true",
                        help="Force cuDNN deterministic kernels (slower).")
    parser.add_argument("--smoke", action="store_true",
                        help="Local pipeline test: 1 epoch / stage, same lr.")
    args = parser.parse_args()

    is_ddp, rank, local_rank, world_size = setup_ddp()
    is_main_proc = is_main(rank)

    # ── Reproducibility: do this BEFORE building model / dataset ──
    set_global_seed(args.seed, rank=rank, deterministic=args.deterministic)
    if is_main_proc:
        print(f"Seed: {args.seed} (deterministic={args.deterministic})")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_proc:
        print(f"Device: {device}, DDP: {is_ddp}, world_size: {world_size}")

    # ── Load configs ──
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    loss_cfg = {}
    if args.loss_config and Path(args.loss_config).exists():
        with open(args.loss_config) as f:
            loss_cfg = yaml.safe_load(f) or {}

    # ── Output dirs ──
    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpt"
    log_dir  = out_dir / "log"
    if is_main_proc:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Snapshot configs alongside checkpoints
        shutil.copy(args.config,      out_dir / "config.yaml")
        if Path(args.loss_config).exists():
            shutil.copy(args.loss_config, out_dir / "loss.yaml")

    # ── Build model + loss ──
    model = CAPModel(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    loss_fn = CAPLoss(cfg=loss_cfg.get("loss", loss_cfg))

    # ── Build data ──
    sh_dim = cfg["gs_param"]["gs_dimension"] - 11   # mu(3)+scale(3)+cov(4)+opacity(1)=11
    dataset = build_dataset(args, sh_dim=sh_dim)
    if is_main_proc:
        print(f"Dataset: {args.dataset!r}  ({len(dataset)} samples)")
    sampler = DistributedSampler(dataset, seed=args.seed) if is_ddp else None
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=_seed_dataloader_worker,
        generator=torch.Generator().manual_seed(args.seed + rank * 10_000),
    )

    # ── Stage schedule (smoke override) ──
    stage_schedule = SMOKE_STAGES if args.smoke else DEFAULT_STAGES
    if args.smoke and is_main_proc:
        print("⚡ SMOKE mode: 1 epoch per stage")

    # ── Resume? ──
    starting_step  = 0
    starting_stage = TrainingStage.RIGID
    if args.resume is not None:
        state = load_checkpoint(Path(args.resume), model,
                                map_location=str(device))
        starting_step  = state.get("global_step", 0)
        starting_stage = state.get("stage", TrainingStage.RIGID)
        if is_main_proc:
            print(f"Resumed from {args.resume} at stage={starting_stage}, "
                  f"step={starting_step}")
        # Skip stages already done
        skip_stages = [s for s in stage_schedule if s.code < starting_stage]
        if state.get("extra", {}).get("stage_done"):
            skip_stages.append(next(s for s in stage_schedule if s.code == starting_stage))
        stages_to_run = [s for s in stage_schedule if s not in skip_stages]
    else:
        stages_to_run = stage_schedule

    # ── Persist seed alongside the run for reproducibility ──
    if is_main_proc:
        (out_dir / "SEED").write_text(f"{args.seed}\n")

    # ── Run stages ──
    global_step = starting_step
    for spec in stages_to_run:
        global_step = run_stage(
            spec=spec, model=model, loss_fn=loss_fn,
            loader=loader, sampler=sampler,
            device=device, ckpt_dir=ckpt_dir, log_dir=log_dir,
            use_amp=(not args.no_amp), grad_clip=args.grad_clip,
            save_every=args.save_every, keep_last=args.keep_last,
            is_main_proc=is_main_proc,
            starting_global_step=global_step,
            is_ddp=is_ddp,
        )

    if is_main_proc:
        # Final convenience symlink
        final = ckpt_dir / "main_exp_final.pt"
        last  = ckpt_dir / "stage_full_done.pt"
        if last.exists():
            if final.exists() or final.is_symlink(): final.unlink()
            try:    final.symlink_to(last.name)
            except OSError: shutil.copy(last, final)
        print(f"\n✓ Training complete.  Final weights: {final}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
