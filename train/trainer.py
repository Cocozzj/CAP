"""
training.py — 4-stage curriculum trainer for CAP main experiment.

Stages (see train/stages.py for full spec; PDF baseline 100ep extended ×1.5 → 150)
─────────────────────────────────────────────────────────────────────────────
  Stage 0  RIGID    — 35 epochs, lr=3e-4   Encoder + Executor (rigid path) trainable;
                                            Planner frozen + forward computed (NCE warmup)
  Stage 1  PLANNER  — 35 epochs, lr=2e-4   Planner CVAE+AR trainable; encoder/executor frozen;
                                            CVAE β + L_comm anneal start here
  Stage 2  PHYSICS  — 25 epochs, lr=1e-4   Executor.deform only; physics ON,
                                            lipschitz + physics_loss on
  Stage 3  FULL     — 55 epochs, lr=3e-5   All trainable, full loss suite,
                                            scheduled sampling 0→0.5 over first 15 ep

Features
────────
  - Multi-GPU via torch.distributed (DDP)
  - Mixed precision (torch.cuda.amp)
  - Gradient clipping (PDF §5.1 explicitly required)
  - TensorBoard logging
  - Checkpoint policy: per-stage end + every N epochs, prune to last K
  - Resume from any checkpoint

Single-GPU::

    python -m train.trainer \\
        --manifest data/dataset_a/manifest.json \\
        --data-dir data/dataset_a/data \\
        --split train --T 30 --image-size 256

Multi-GPU launch (8 GPUs)::

    torchrun --nproc_per_node=8 -m train.trainer \\
        --manifest data/dataset_a/manifest.json \\
        --data-dir data/dataset_a/data \\
        --batch-size 8 --num-workers 4 --auto-test

Resume::

    python -m train.trainer --resume runs/exp1/ckpt/stage_rigid_done.pt \\
        --manifest data/dataset_a/manifest.json \\
        --data-dir data/dataset_a/data
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import yaml

from model import CAPModel, CAPLoss
from dataload import DatasetA, DatasetB, collate_batch
from .stages import StageSpec, DEFAULT_STAGES, SMOKE_STAGES


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
    stage_name: str,
    epoch: int,
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint atomically (write to .tmp then rename).

    ``stage_name`` is the StageSpec.name string ("RIGID" / "PHYSICS" / ...).
    Stored as a string so resume doesn't depend on a fixed integer enum.
    """
    state = {
        "model":       (model.module if hasattr(model, "module") else model).state_dict(),
        "optimizer":   optimizer.state_dict(),
        "scaler":      scaler.state_dict() if scaler is not None else None,
        "stage_name":  stage_name,
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
    """Build Adam over only the trainable parameters in their current state.

    set_trainable() flips requires_grad on/off, so we filter here per-call.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# ══════════════════════════════════════════════════════════════════════
# One epoch
# ══════════════════════════════════════════════════════════════════════

def _step(model, optimizer, scaler, total, grad_clip):
    """Backward + grad-clip + optimizer step.  AMP-aware (single path)."""
    if scaler is not None:
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
    else:
        total.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=grad_clip,
    )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()


def _log(losses, total, lr, global_step, spec_name, epoch, writer):
    """Print one log line + dump all loss tensors to TensorBoard."""
    print(f"[stage={spec_name} epoch={epoch} step={global_step}] "
          f"total={total.item():.4f}", flush=True)
    if writer is not None:
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                writer.add_scalar(f"loss/{k}", v.item(), global_step)
        writer.add_scalar("lr", lr, global_step)


def _compute_tau(global_step: int, tau_sched: Dict[str, Any], steps_per_epoch: int) -> float:
    """Linear Gumbel-softmax temperature anneal driven by cumulative global_step.

    yaml: training.tau_schedule = {start, end, anneal_epochs}.
    Falls back to start (no anneal) if anneal_epochs<=0 or schedule missing.
    """
    if not tau_sched:
        return 1.0
    start         = float(tau_sched.get("start", 1.0))
    end           = float(tau_sched.get("end",   1.0))
    anneal_epochs = int(tau_sched.get("anneal_epochs", 0))
    anneal_steps  = anneal_epochs * max(steps_per_epoch, 1)
    if anneal_steps <= 0:
        return start
    frac = min(global_step / float(anneal_steps), 1.0)
    return start + (end - start) * frac


def train_epoch(
    *,
    model:     nn.Module,
    loss_fn:   CAPLoss,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler:    Optional[GradScaler],
    device:    torch.device,
    spec:      StageSpec,
    epoch:     int,
    global_step: int,
    stage_start_global_step: int,         # P0-2: anneal uses stage-local step
    total_steps: int,                     # per-stage iters; loss anneal denominator
    tau_schedule: Optional[Dict[str, Any]] = None,   # P2-5: Gumbel temp anneal
    steps_per_epoch: int = 1,
    grad_clip: float = 1.0,
    log_every: int = 10,
    writer:    Optional[SummaryWriter] = None,
    is_main_proc: bool = True,
) -> int:
    """Run one training epoch, return updated global_step."""
    base_model = model.module if hasattr(model, "module") else model
    # NOTE: do NOT call ``model.train()`` here — _prepare_stage already called
    # ``set_trainable(...)`` which puts frozen submodules into eval mode.  A
    # blanket ``model.train()`` would re-enable train-mode side-effects (VQ
    # codebook EMA, BN running stats, etc.) on supposedly-frozen modules.
    fwd_ctx    = autocast if scaler is not None else contextlib.nullcontext

    # Scheduled-sampling ramp inside this stage.  PDF Stage-2 wants
    # sample_prob to rise so AR decoder gets used to its own tokens at
    # train time (mitigates exposure bias).
    if spec.sample_prob_ramp_epochs > 0:
        frac        = min(epoch / float(spec.sample_prob_ramp_epochs), 1.0)
        sample_prob = spec.sample_prob_max * frac
    else:
        sample_prob = 0.0

    for batch_idx, batch in enumerate(loader):
        frames    = batch["frames"].to(device, non_blocking=True)
        gs_params = [g.to(device=device) for g in batch["gs_params"]]
        # condition carries text + sample_prob + render density (per-stage).
        # render_n_timesteps=0 in PLANNER stage skips rendering entirely.
        condition = {
            "texts":             batch.get("text"),
            "sample_prob":       sample_prob,
            "render_n_timesteps": spec.render_n_timesteps,
        }

        # Cameras → renderer → rec_loss.  Without this the rec/lpips/depth
        # losses early-exit to 0 (loss.py:reconstruction_loss).
        cameras = None
        if "intrinsics" in batch and "extrinsics" in batch:
            cameras = {
                "intrinsics": batch["intrinsics"].to(device, non_blocking=True),
                "extrinsics": batch["extrinsics"].to(device, non_blocking=True),
            }

        # P2-5: Gumbel temperature anneal driven by cumulative global_step
        tau = _compute_tau(global_step, tau_schedule or {}, steps_per_epoch)

        optimizer.zero_grad(set_to_none=True)

        with fwd_ctx():
            training_out = model(
                frames, gs_params=gs_params,
                enable_physics=spec.enable_physics,
                run_planner=spec.run_planner,
                tau=tau, condition=condition, cameras=cameras,
            )
            # P0-1: pass the input frames as GT — autoencoder reconstructs back to
            # the same frames; without this rec_mse / lpips / depth all early-exit
            # to 0 (loss.py:429) and visual reconstruction never trains.
            # P0-2: loss anneal must use stage-local step so frac sweeps 0→1
            # within each stage, not jumps to 1 immediately at stage transitions.
            losses = loss_fn(
                model=base_model, training_out=training_out,
                gt={"frames": frames,
                    "depth":  batch.get("depth").to(device, non_blocking=True)
                              if batch.get("depth") is not None else None,
                    "text":   condition["texts"]},
                spec=spec.loss,
                step=global_step - stage_start_global_step,
                total_steps=total_steps,
            )
            total = losses["total"]

        # NaN diagnostic: when total is NaN, dump every per-component loss
        # value so we can localise WHICH term blew up.  Fires only on first
        # NaN per epoch to avoid log spam.
        if is_main_proc and torch.isnan(total).any() and not getattr(train_epoch, "_nan_reported", False):
            print(f"\n  ⚠ NaN detected at stage={spec.name} step={global_step} — per-component breakdown:")
            for k, v in sorted(losses.items()):
                if isinstance(v, torch.Tensor):
                    val = v.item() if v.numel() == 1 else v.float().mean().item()
                    flag = " ← NaN" if (val != val) else ""
                    print(f"      {k:20s} = {val:+.4f}{flag}")
            train_epoch._nan_reported = True

        _step(model, optimizer, scaler, total, grad_clip)
        global_step += 1

        if is_main_proc and (batch_idx % log_every == 0):
            _log(losses, total, optimizer.param_groups[0]["lr"],
                 global_step, spec.name, epoch, writer)
            if writer is not None:
                writer.add_scalar("schedule/tau",         tau,         global_step)
                writer.add_scalar("schedule/sample_prob", sample_prob, global_step)

    return global_step


# ══════════════════════════════════════════════════════════════════════
# Validation loop — measures stage-internal val_loss for best-ckpt tracking
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_val(
    base_model: nn.Module,
    val_loader: DataLoader,
    loss_fn:    CAPLoss,
    spec:       StageSpec,
    device:     torch.device,
    *,
    stage_step: int,
    total_iters: int,
    tau:        float = 0.1,
) -> float:
    """Compute mean per-sample val loss using ``spec.loss`` (same anneal state
    as train at this point).  Called from main rank only — bypasses DDP wrapper
    by calling ``base_model`` directly.

    Returns total loss averaged per sample (NOT per batch).  Always uses
    ``sample_prob=0`` (pure teacher forcing) for a stable measurement.
    """
    # Snapshot per-submodule train/eval state so we can restore after val.
    # We can't just toggle base_model.train()/eval() because some submodules
    # are intentionally frozen (in eval) for the current stage — see
    # CAPModel.set_trainable.
    saved_modes = {n: m.training for n, m in base_model.named_modules()}
    base_model.eval()
    total_sum = 0.0
    n_samples = 0
    for batch in val_loader:
        frames    = batch["frames"].to(device, non_blocking=True)
        gs_params = [g.to(device=device) for g in batch["gs_params"]]
        # Mirror train: same render density so val_loss is comparable to train_loss
        condition = {
            "texts":              batch.get("text"),
            "sample_prob":        0.0,
            "render_n_timesteps": spec.render_n_timesteps,
        }

        cameras = None
        if "intrinsics" in batch and "extrinsics" in batch:
            cameras = {
                "intrinsics": batch["intrinsics"].to(device, non_blocking=True),
                "extrinsics": batch["extrinsics"].to(device, non_blocking=True),
            }

        training_out = base_model(
            frames, gs_params=gs_params,
            enable_physics=spec.enable_physics,
            run_planner=spec.run_planner,
            tau=tau, condition=condition, cameras=cameras,
        )
        losses = loss_fn(
            model=base_model, training_out=training_out,
            gt={"frames": frames,
                "depth":  batch.get("depth").to(device, non_blocking=True)
                          if batch.get("depth") is not None else None,
                "text":   condition["texts"]},
            spec=spec.loss, step=stage_step, total_steps=total_iters,
        )
        B = frames.size(0)
        total_sum += losses["total"].item() * B
        n_samples += B

    # Restore the exact per-submodule train/eval state we captured above.
    for n, m in base_model.named_modules():
        if n in saved_modes:
            m.training = saved_modes[n]
    return total_sum / max(n_samples, 1)


# ══════════════════════════════════════════════════════════════════════
# Stage runner
# ══════════════════════════════════════════════════════════════════════

def _prepare_stage(
    spec, base_model, log_dir, use_amp, is_main_proc,
    *,
    is_ddp:             bool,
    local_rank:         int,
    resume_optim_state: Optional[Dict[str, Any]] = None,
    resume_scaler_state: Optional[Dict[str, Any]] = None,
):
    """Apply trainable flags + (re-)wrap DDP + build optimizer/scaler/writer.

    P2-6: re-wrap DDP per stage so we don't carry ``find_unused_parameters=True``
    (perf overhead) into FULL stage where everything is trainable.  Re-wrapping
    is safe: DDP doesn't copy parameters, it only registers grad hooks.
    """
    base_model.set_trainable(
        encoder=spec.encoder, planner=spec.planner,
        executor=spec.executor, deform_only=spec.deform_only,
    )

    # Wrap with DDP iff distributed.  Stages that freeze submodules need
    # find_unused_parameters; FULL stage doesn't.
    if is_ddp:
        any_frozen = (not spec.encoder) or (not spec.planner) or \
                     (not spec.executor) or spec.deform_only
        model = DDP(base_model, device_ids=[local_rank],
                    find_unused_parameters=any_frozen)
    else:
        model = base_model

    optimizer = build_optimizer(model, spec.lr)
    scaler    = GradScaler() if (use_amp and torch.cuda.is_available()) else None
    writer    = SummaryWriter(log_dir / spec.name) if is_main_proc else None

    # P1-4: restore optimizer/scaler state ONLY when resuming mid-stage (caller
    # passes None for fresh stage starts so transitions get clean Adam moments).
    if resume_optim_state is not None:
        try:
            optimizer.load_state_dict(resume_optim_state)
            if is_main_proc:
                print(f"  ↻ restored optimizer state for resume")
        except (ValueError, KeyError) as e:
            if is_main_proc:
                print(f"  ⚠ optimizer state mismatch ({e}) — starting fresh")
    if resume_scaler_state is not None and scaler is not None:
        try:
            scaler.load_state_dict(resume_scaler_state)
        except (ValueError, KeyError):
            pass

    if is_main_proc:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== Stage {spec.name} ===")
        print(f"  epochs={spec.epochs}, lr={spec.lr}, "
              f"physics={spec.enable_physics}, trainable={n_train:,}"
              + (f"  [DDP find_unused={any_frozen}]" if is_ddp else ""))

    return model, optimizer, scaler, writer


def run_stage(
    *,
    spec:         StageSpec,
    base_model:   nn.Module,                              # P2-6: un-wrapped CAPModel
    loss_fn:      CAPLoss,
    loader:       DataLoader,
    sampler:      Optional[DistributedSampler],
    val_loader:   Optional[DataLoader],                   # best-val tracking (None → off)
    val_every:    int,
    device:       torch.device,
    ckpt_dir:     Path,
    log_dir:      Path,
    use_amp:      bool,
    grad_clip:    float,
    save_every:   int,
    keep_last:    int,
    is_main_proc: bool,
    starting_global_step:    int,
    starting_epoch_in_stage: int,                         # P1-3: mid-stage resume
    resume_optim_state:      Optional[Dict[str, Any]],    # P1-4
    resume_scaler_state:     Optional[Dict[str, Any]],    # P1-4
    is_ddp:       bool,
    local_rank:   int,
    tau_schedule: Optional[Dict[str, Any]] = None,        # P2-5
) -> int:
    """Train one stage, return the cumulative global_step at end."""
    model, optimizer, scaler, writer = _prepare_stage(
        spec, base_model, log_dir, use_amp, is_main_proc,
        is_ddp=is_ddp, local_rank=local_rank,
        resume_optim_state=resume_optim_state,
        resume_scaler_state=resume_scaler_state,
    )
    steps_per_epoch         = len(loader)
    total_iters             = spec.epochs * steps_per_epoch     # per-stage anneal denom
    stage_start_global_step = starting_global_step              # P0-2: anneal reference
    global_step             = starting_global_step

    # Per-stage best-val tracking (val losses are NOT comparable across stages)
    best_val      = float("inf")
    best_val_path = ckpt_dir / f"best_val_{spec.name.lower()}.pt"

    # P1-3: resume picks up where we left off; for fresh starts this is 0
    for epoch in range(starting_epoch_in_stage, spec.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        t0 = time.time()
        global_step = train_epoch(
            model=model, loss_fn=loss_fn, loader=loader,
            optimizer=optimizer, scaler=scaler,
            device=device, spec=spec, epoch=epoch,
            global_step=global_step,
            stage_start_global_step=stage_start_global_step,
            total_steps=total_iters,
            tau_schedule=tau_schedule, steps_per_epoch=steps_per_epoch,
            grad_clip=grad_clip, writer=writer, is_main_proc=is_main_proc,
        )
        if is_main_proc:
            dt = time.time() - t0
            print(f"  epoch {epoch + 1}/{spec.epochs} done in {dt:.1f}s  "
                  f"step={global_step}", flush=True)
            if (epoch + 1) % save_every == 0:
                p = ckpt_dir / f"epoch_{global_step:08d}.pt"
                save_checkpoint(p, model, optimizer, scaler,
                                stage_name=spec.name, epoch=epoch + 1,
                                global_step=global_step, extra={})
                prune_checkpoints(ckpt_dir, keep_last=keep_last)
                print(f"  ✔ saved {p.name}")

        # ── Validation pass (main rank only; others wait at barrier) ──
        if val_loader is not None and ((epoch + 1) % val_every == 0):
            if is_main_proc:
                stage_step = global_step - stage_start_global_step
                val_loss = run_val(
                    base_model, val_loader, loss_fn, spec, device,
                    stage_step=stage_step, total_iters=total_iters,
                )
                print(f"  val_loss={val_loss:.4f}  (best={min(best_val, val_loss):.4f})",
                      flush=True)
                if writer is not None:
                    writer.add_scalar("val/total", val_loss, global_step)
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        best_val_path, model, optimizer, scaler,
                        stage_name=spec.name, epoch=epoch + 1,
                        global_step=global_step,
                        extra={"val_loss": val_loss, "best_val": True},
                    )
                    print(f"  ✨ new best → {best_val_path.name}")
            if is_ddp:
                dist.barrier()   # other ranks wait until main rank finishes val

    if is_main_proc:
        p = ckpt_dir / f"stage_{spec.name.lower()}_done.pt"
        save_checkpoint(p, model, optimizer, scaler,
                        stage_name=spec.name, epoch=spec.epochs,
                        global_step=global_step,
                        extra={"stage_done": True, "best_val": best_val})
        print(f"  ★ stage-end ckpt: {p.name}  best_val={best_val:.4f}")
        if writer is not None:
            writer.close()

    if is_ddp:
        dist.barrier()
        # Free the per-stage DDP wrapper before re-wrapping in next stage —
        # avoids accumulating Reducer state across the curriculum.
        del model
    return global_step


# ══════════════════════════════════════════════════════════════════════
# Dataset factory
# ══════════════════════════════════════════════════════════════════════

def build_dataset(args, sh_dim: int, *, split: Optional[str] = None):
    """Build a Dataset instance based on ``args.dataset`` (a or b).

    ``split`` overrides ``args.split`` so the same factory can build train,
    val, and test sets without duplicating arg-handling.
    """
    cls = DatasetA if args.dataset == "a" else DatasetB
    return cls(
        manifest_path = args.manifest,
        data_dir      = args.data_dir,
        split         = split if split is not None else args.split,
        T             = args.T,
        image_size    = args.image_size,
        c_sh          = sh_dim,
    )


def _build_val_loader(args, sh_dim: int, is_main_proc: bool) -> Optional[DataLoader]:
    """Build val DataLoader.  Only main rank uses it (others wait at barrier),
    so no DistributedSampler — main rank sees the full val set.

    Returns None if ``--no-val`` or if the val split doesn't exist.
    """
    if args.no_val:
        return None
    try:
        ds = build_dataset(args, sh_dim=sh_dim, split=args.val_split)
    except (ValueError, SystemExit) as e:
        if is_main_proc:
            print(f"⚠ val split {args.val_split!r} unavailable ({e}) — best-val tracking disabled")
        return None
    if is_main_proc:
        print(f"Val:     {args.val_split!r}  ({len(ds)} samples)")
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, sampler=None,
        num_workers=args.num_workers, collate_fn=collate_batch,
        pin_memory=True, drop_last=False,
    )


# ══════════════════════════════════════════════════════════════════════
# Main — orchestrator + 6 helpers
# ══════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config",      type=str, default="configs/config.yaml")
    p.add_argument("--loss-config", type=str, default="configs/loss.yaml")
    p.add_argument("--out-dir",     type=str, default="results/main_exp")
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-amp",      action="store_true")
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--save-every",  type=int, default=5)
    p.add_argument("--keep-last",   type=int, default=3)
    p.add_argument("--resume",      type=str, default=None)
    # Dataset
    p.add_argument("--dataset",       type=str, default="a", choices=["a", "b"],
                   help="a = DatasetA (3-cam synthetic), b = DatasetB (1-cam real video)")
    p.add_argument("--manifest",      type=str, required=True,
                   help="data/dataset_<a|b>/manifest.json")
    p.add_argument("--data-dir",      type=str, required=True,
                   help="data/dataset_<a|b>/data")
    p.add_argument("--split",         type=str, default="train")
    p.add_argument("--T",             type=int, default=30,
                   help="frames per sample (>=10, %%5==0)")
    p.add_argument("--image-size",    type=int, default=256)
    # Validation + auto-test
    p.add_argument("--val-split", type=str, default="val",
                   help="dataset split name to use as val (default: val)")
    p.add_argument("--val-every", type=int, default=1,
                   help="run val every N epochs (default: 1 = every epoch)")
    p.add_argument("--no-val",    action="store_true",
                   help="disable val loop entirely (no best_val_*.pt tracking)")
    p.add_argument("--auto-test", action="store_true",
                   help="after training: load main_exp_final.pt and eval on --test-splits")
    p.add_argument("--test-splits", type=str, nargs="+",
                   default=["test_iid", "test_ood_unseen_pair",
                            "test_ood_unseen_object", "test_compositional_long"],
                   help="splits to evaluate on when --auto-test is set")
    # Reproducibility / mode
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--smoke",         action="store_true",
                   help="1 epoch / stage, same lr — local pipeline test")
    return p.parse_args()


def _setup_run(args) -> Tuple[bool, int, int, torch.device, bool]:
    """Init DDP + seed + device.  Returns (is_ddp, rank, local_rank, device, is_main_proc)."""
    is_ddp, rank, local_rank, world_size = setup_ddp()
    is_main_proc = is_main(rank)
    set_global_seed(args.seed, rank=rank, deterministic=args.deterministic)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_proc:
        print(f"Seed: {args.seed} (det={args.deterministic})  "
              f"Device: {device}  DDP: {is_ddp}  world: {world_size}")
    return is_ddp, rank, local_rank, device, is_main_proc


def _load_configs(args) -> Tuple[dict, dict]:
    """Load main + loss configs.  Returns (cfg, loss_cfg)."""
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    loss_cfg = {}
    if args.loss_config and Path(args.loss_config).exists():
        with open(args.loss_config) as f:
            loss_cfg = yaml.safe_load(f) or {}
    return cfg, loss_cfg


def _check_renderer_sanity(loss_cfg: dict, is_main_proc: bool) -> None:
    """Detect the silent-failure case: yaml asks for rec/lpips/depth losses
    but the gsplat renderer isn't available → exec_out["rendered_frames"]
    will always be None → reconstruction_loss early-exits to 0.

    Print a LOUD warning so the user catches misconfigured installs before
    spending hours training without any pixel-level supervision.
    """
    if not is_main_proc:
        return
    cfg = loss_cfg.get("loss", loss_cfg) or {}
    rec_weights = {
        "lambda_rec":       float(cfg.get("lambda_rec",       0.0)),
        "lambda_rec_mse":   float(cfg.get("lambda_rec_mse",   0.0)),
        "lambda_rec_lpips": float(cfg.get("lambda_rec_lpips", 0.0)),
        "lambda_depth":     float(cfg.get("lambda_depth",     0.0)),
    }
    asks_rec = any(w > 0 for w in rec_weights.values())
    if not asks_rec:
        return
    try:
        from model.executor.renderer import gsplat_available
        ok = gsplat_available()
    except Exception:
        ok = False
    if not ok:
        active = ", ".join(f"{k}={v}" for k, v in rec_weights.items() if v > 0)
        print("\n" + "=" * 70)
        print("⚠ WARNING: reconstruction loss is configured but gsplat is unavailable")
        print(f"  Active loss weights: {active}")
        print(f"  → exec_out['rendered_frames'] will be None for every batch")
        print(f"  → reconstruction_loss returns 0; rec/lpips/depth contribute NOTHING")
        print(f"  → model trains on algebraic + InfoNCE + VQ + planner signals only")
        print(f"  Fix: `pip install gsplat`  OR set lambda_rec*/lambda_depth=0 in loss.yaml")
        print("=" * 70 + "\n", flush=True)


def _setup_run_dirs(args, is_main_proc) -> Tuple[Path, Path, Path]:
    """Create out / ckpt / log dirs, snapshot configs.  Returns (out_dir, ckpt_dir, log_dir)."""
    out_dir  = Path(args.out_dir)
    ckpt_dir = out_dir / "ckpt"
    log_dir  = out_dir / "log"
    if is_main_proc:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.config, out_dir / "config.yaml")
        if Path(args.loss_config).exists():
            shutil.copy(args.loss_config, out_dir / "loss.yaml")
        (out_dir / "SEED").write_text(f"{args.seed}\n")
    return out_dir, ckpt_dir, log_dir


def _build_model_and_data(args, cfg, loss_cfg, is_ddp, rank, device, is_main_proc):
    """Instantiate model + loss + dataset + loader.

    Returns the **un-wrapped** CAPModel; DDP wrapping is deferred to
    ``_prepare_stage`` so each stage gets the right ``find_unused_parameters``
    setting (P2-6).  ``local_rank`` is no longer needed here since DDP wrap
    moved out — caller passes it directly to ``run_stage``.
    """
    model = CAPModel(cfg).to(device)
    loss_fn = CAPLoss(cfg=loss_cfg.get("loss", loss_cfg))

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    dataset = build_dataset(args, sh_dim=sh_dim)
    if is_main_proc:
        print(f"Train:   {args.split!r}  ({len(dataset)} samples)")

    sampler = DistributedSampler(dataset, seed=args.seed) if is_ddp else None
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate_batch,
        pin_memory=True, drop_last=True,
        worker_init_fn=_seed_dataloader_worker,
        generator=torch.Generator().manual_seed(args.seed + rank * 10_000),
    )
    val_loader = _build_val_loader(args, sh_dim=sh_dim, is_main_proc=is_main_proc)
    return model, loss_fn, loader, sampler, val_loader


def _resolve_resume(
    args, model, device, stage_schedule, is_main_proc,
) -> Tuple[List[StageSpec], int, int, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Apply --resume.

    Returns:
      stages_to_run             — slice of curriculum from the resume point
      starting_global_step      — cumulative step counter at resume
      starting_epoch_in_stage   — P1-3: how many epochs of the resumed stage are done
      resume_optim_state        — P1-4: optimizer state dict if mid-stage, else None
      resume_scaler_state       — P1-4: GradScaler state dict if present, else None
    """
    if args.resume is None:
        return list(stage_schedule), 0, 0, None, None

    # Load model weights via existing helper; pull raw state for optim/scaler.
    state = load_checkpoint(Path(args.resume), model, map_location=str(device))
    starting_step      = state.get("global_step", 0)
    resumed_stage_name = state.get("stage_name", stage_schedule[0].name)
    stage_done         = bool(state.get("extra", {}).get("stage_done"))
    epochs_completed   = int(state.get("epoch", 0))   # epochs done within this stage

    names = [s.name for s in stage_schedule]
    if resumed_stage_name not in names:
        raise SystemExit(f"Resume stage {resumed_stage_name!r} not in {names}")
    idx = names.index(resumed_stage_name)

    if stage_done:
        # Move to next stage with fresh optimizer (intentional reset on transition)
        start_idx, start_epoch, optim_state, scaler_state = idx + 1, 0, None, None
    else:
        # Mid-stage resume: continue same stage with restored optimizer
        start_idx    = idx
        start_epoch  = epochs_completed
        optim_state  = state.get("optimizer")
        scaler_state = state.get("scaler")

    if is_main_proc:
        kind = "stage-end → next stage" if stage_done else f"mid-stage → epoch {start_epoch}"
        print(f"Resumed from {args.resume}  stage={resumed_stage_name!r}  "
              f"step={starting_step}  ({kind})")

    return list(stage_schedule[start_idx:]), starting_step, start_epoch, optim_state, scaler_state


def _link_or_copy(link_path: Path, target_name: str) -> None:
    """Atomically point ``link_path`` to a sibling ``target_name``.
    Symlink first; falls back to file copy on filesystems that disallow it
    (Windows w/o admin, some NFS / Docker volumes).
    """
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    try:
        link_path.symlink_to(target_name)
    except OSError:
        shutil.copy(link_path.parent / target_name, link_path)


def _finalize(ckpt_dir, is_main_proc):
    """Create two top-level entry points after training:

      main_exp_final.pt → best_val_full.pt   (best on val — for downstream eval / deploy)
      main_exp_last.pt  → stage_full_done.pt (deterministic endpoint — for paper repro)

    Falls back to ``stage_full_done.pt`` for ``main_exp_final`` if no val
    tracking happened (e.g. ``--no-val`` or val split missing).
    """
    if not is_main_proc:
        return

    last_src  = ckpt_dir / "stage_full_done.pt"
    best_src  = ckpt_dir / "best_val_full.pt"
    last_link = ckpt_dir / "main_exp_last.pt"
    final_link= ckpt_dir / "main_exp_final.pt"

    if last_src.exists():
        _link_or_copy(last_link, last_src.name)

    if best_src.exists():
        _link_or_copy(final_link, best_src.name)
    elif last_src.exists():
        # No val tracking — fall back so downstream eval scripts don't break
        _link_or_copy(final_link, last_src.name)

    print(f"\n✓ Training complete.")
    print(f"  Final (best-val):  {final_link}")
    print(f"  Last  (FULL end):  {last_link}")


def _run_auto_test(args, base_model, ckpt_dir, out_dir, cfg, loss_cfg,
                   device, is_main_proc, is_ddp):
    """Load best-val checkpoint and evaluate on every test split.  Writes
    ``test_results.json`` next to the ckpt dir.  Main rank only — others wait.
    """
    if is_ddp:
        dist.barrier()
    if not is_main_proc:
        return

    ckpt_path = ckpt_dir / "main_exp_final.pt"
    if not ckpt_path.exists():
        print("  ⚠ no main_exp_final.pt — skipping auto-test")
        return

    print(f"\n=== Auto-test: loading {ckpt_path.name} ===")
    state = torch.load(ckpt_path, map_location=str(device))
    target = base_model.module if hasattr(base_model, "module") else base_model
    target.load_state_dict(state["model"])

    # Use FULL stage's loss flags for test (matches eval-time loss composition)
    full_spec = DEFAULT_STAGES[-1]
    loss_fn   = CAPLoss(cfg=loss_cfg.get("loss", loss_cfg))
    sh_dim    = cfg["gs_param"]["gs_dimension"] - 11

    results: Dict[str, Any] = {
        "checkpoint":   str(ckpt_path),
        "loaded_from":  state.get("stage_name"),
        "loaded_step":  state.get("global_step"),
        "splits":       {},
    }

    for split_name in args.test_splits:
        try:
            ds = build_dataset(args, sh_dim=sh_dim, split=split_name)
        except (ValueError, SystemExit) as e:
            print(f"  skip {split_name:30s} ({e})")
            continue
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, sampler=None,
            num_workers=args.num_workers, collate_fn=collate_batch,
            pin_memory=True, drop_last=False,
        )
        # stage_step=total_iters → anneal frac=1 → full-strength loss weights
        val_loss = run_val(
            target, loader, loss_fn, full_spec, device,
            stage_step=1, total_iters=1,
        )
        results["splits"][split_name] = {"loss": val_loss, "n": len(ds)}
        print(f"  {split_name:30s}  loss={val_loss:.4f}  n={len(ds)}")

    out_path = out_dir / "test_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"  ✓ saved {out_path}")


def main():
    args = _parse_args()
    is_ddp, rank, local_rank, device, is_main_proc = _setup_run(args)
    cfg, loss_cfg = _load_configs(args)
    _check_renderer_sanity(loss_cfg, is_main_proc)
    out_dir, ckpt_dir, log_dir = _setup_run_dirs(args, is_main_proc)
    base_model, loss_fn, loader, sampler, val_loader = _build_model_and_data(
        args, cfg, loss_cfg, is_ddp, rank, device, is_main_proc,
    )

    stage_schedule = SMOKE_STAGES if args.smoke else DEFAULT_STAGES
    if args.smoke and is_main_proc:
        print("⚡ SMOKE mode: 1 epoch per stage")

    # P2-5: Gumbel-softmax temperature anneal config (yaml: training.tau_schedule)
    tau_schedule = (loss_cfg.get("training", {}) or {}).get("tau_schedule")

    stages_to_run, global_step, start_epoch, optim_state, scaler_state = _resolve_resume(
        args, base_model, device, stage_schedule, is_main_proc,
    )

    # Resume state (epoch + optimizer + scaler) only applies to the FIRST stage
    # in stages_to_run; subsequent stages start fresh.
    for i, spec in enumerate(stages_to_run):
        global_step = run_stage(
            spec=spec, base_model=base_model, loss_fn=loss_fn,
            loader=loader, sampler=sampler,
            val_loader=val_loader, val_every=args.val_every,
            device=device, ckpt_dir=ckpt_dir, log_dir=log_dir,
            use_amp=(not args.no_amp), grad_clip=args.grad_clip,
            save_every=args.save_every, keep_last=args.keep_last,
            is_main_proc=is_main_proc,
            starting_global_step=global_step,
            starting_epoch_in_stage=(start_epoch  if i == 0 else 0),
            resume_optim_state    =(optim_state  if i == 0 else None),
            resume_scaler_state   =(scaler_state if i == 0 else None),
            is_ddp=is_ddp, local_rank=local_rank,
            tau_schedule=tau_schedule,
        )

    _finalize(ckpt_dir, is_main_proc)
    if args.auto_test:
        _run_auto_test(args, base_model, ckpt_dir, out_dir, cfg, loss_cfg,
                       device, is_main_proc, is_ddp)
    cleanup_ddp()


if __name__ == "__main__":
    main()
