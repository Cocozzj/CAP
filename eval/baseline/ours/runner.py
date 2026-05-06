"""Run Ours' inference on test splits, write pred_4dgs.npz per trajectory.

This adapter is the bridge between our trained CAPModel and the unified
baseline aggregator.  After this runs, ``runs/baselines/ours/<dataset>/<split>/<traj_id>/``
contains the same ``pred_4dgs.npz`` + ``metrics.json`` schema as TAMP-rule,
PhysGaussian, etc., so ``aggregate.py`` automatically includes Ours in the
main table.

Pipeline per trajectory:
    1. Load init_gs.ply + cameras + text
    2. model.infer_text(text, scene)  → trajectory: List[SceneState] of length T
    3. Stack the trajectory's mu/cov/sh/opacity/scale into [T, N, ...] arrays
    4. Save as GS4DSequence (.npz)

For diversity (PDF #9), this script also exposes ``sample_action_tokens()``
which is consumed by eval/baseline/diversity_eval.py.

Usage:

    python -m eval.baseline.ours.runner \\
        --ckpt runs/main_exp/seed_0/ckpt/main_exp_final.pt \\
        --config configs/config.yaml \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid test_ood_unseen_pair test_ood_unseen_object test_compositional_long \\
        --output-root runs/baselines \\
        --dataset-name dataset_a
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from dataload.common import load_cameras, load_init_gs_ply
from dataload.text import task_to_text
from model.model import CAPModel
from model.utils import SceneState

from ..common import (
    GS4DSequence,
    TrajMetrics,
    baseline_output_dir,
    iter_split_entries,
)


# ══════════════════════════════════════════════════════════════════════
# Model loading (cached)
# ══════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: Path | str, config_path: Path | str,
                device: torch.device) -> CAPModel:
    """Load Ours' CAPModel from a checkpoint, with sentence-transformers
    key remapping.

    sentence-transformers' internal Transformer module renamed its
    attribute ``self.model`` (older versions) to ``self.auto_model``
    (newer versions).  All ~199 language-encoder keys differ only by
    that single component:

        planner.lang.text_enc.model.0.model.*       ← old (training-time)
        planner.lang.text_enc.model.0.auto_model.*  ← new (inference-time)

    We strict-rename the ckpt keys so load_state_dict succeeds.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model = CAPModel(cfg).to(device)
    state = torch.load(str(ckpt_path), map_location=str(device))
    raw_sd = state["model"]

    # Detect which side has the obsolete name and remap.
    target_keys = set(model.state_dict().keys())
    has_old = any(".model.0.model." in k for k in raw_sd.keys())
    needs_new = any(".model.0.auto_model." in k for k in target_keys)
    if has_old and needs_new:
        remapped = {}
        n = 0
        for k, v in raw_sd.items():
            if ".model.0.model." in k:
                k2 = k.replace(".model.0.model.", ".model.0.auto_model.")
                n += 1
            else:
                k2 = k
            remapped[k2] = v
        if n:
            print(f"  ⏬ remapped {n} sentence-transformers keys "
                  f"(model → auto_model)")
        raw_sd = remapped

    msg = model.load_state_dict(raw_sd, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"  ⚠ load_state_dict — missing: {len(msg.missing_keys)}, "
              f"unexpected: {len(msg.unexpected_keys)}")
        # Show first few of each so we can spot any *other* mismatch
        if msg.missing_keys:
            print(f"    missing[:3]:    {msg.missing_keys[:3]}")
        if msg.unexpected_keys:
            print(f"    unexpected[:3]: {msg.unexpected_keys[:3]}")
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════
# Build a single-trajectory SceneState from init_gs.ply + cameras.json
# ══════════════════════════════════════════════════════════════════════

def _build_initial_scene(
    traj_dir: Path,
    n_gs_points: int,
    c_sh: int,
    n_slots: int,
    device: torch.device,
) -> SceneState:
    """Construct a B=1, K=n_slots SceneState from init_gs.ply.

    For inference we only need the geometric / color fields populated; the
    canonical frame (phi) is initialized to identity, the assignment puts all
    Gaussians in slot 0.  This matches how train_epoch builds scenes when
    only init_gs is available.
    """
    gs = load_init_gs_ply(traj_dir / "init_gs.ply",
                           n_points=n_gs_points, seed=0, c_sh=c_sh)

    N = int(gs.mu.shape[0])
    K = int(n_slots)
    B = 1

    mu0      = gs.mu.to(device).float()              # [N, 3]
    sh0      = gs.sh.to(device).float()              # [N, c_sh]
    opacity0 = gs.opacity.to(device).float()         # [N, 1]
    scale0   = gs.scale.to(device).float()           # [N, 3]

    # Convert (quat, log_scale) → full 3×3 cov
    quat = gs.cov.to(device).float()                 # [N, 4]   xyzw or wxyz
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    R = torch.stack([
        torch.stack([1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)], -1),
        torch.stack([    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)], -1),
        torch.stack([    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)], -1),
    ], -2).float()                                    # [N, 3, 3]
    s = torch.exp(scale0)                             # [N, 3]
    S = torch.zeros_like(R)
    S[..., 0, 0] = s[..., 0]; S[..., 1, 1] = s[..., 1]; S[..., 2, 2] = s[..., 2]
    cov0 = R @ S @ S @ R.transpose(-1, -2)            # [N, 3, 3]

    # Pad N → N_max per slot.  Simplest: all Gaussians in slot 0,
    # other slots empty (mask=False, opacity=0).
    N_max = N
    mu_padded      = torch.zeros(B, K, N_max, 3,    device=device)
    cov_padded     = torch.zeros(B, K, N_max, 3, 3, device=device)
    sh_padded      = torch.zeros(B, K, N_max, sh0.shape[-1], device=device)
    opacity_padded = torch.zeros(B, K, N_max, 1,    device=device)
    scale_padded   = torch.zeros(B, K, N_max, 3,    device=device)
    mask = torch.zeros(B, K, N_max, dtype=torch.bool, device=device)

    # Put everything in slot 0
    mu_padded[0, 0]      = mu0
    cov_padded[0, 0]     = cov0
    sh_padded[0, 0]      = sh0
    opacity_padded[0, 0] = opacity0
    scale_padded[0, 0]   = scale0
    mask[0, 0]           = True

    # Identity canonical frame Φ_o per slot
    from model.utils import CanonicalFrame
    R_w2c = torch.eye(3, device=device).expand(B, K, 3, 3).contiguous()
    t_w2c = torch.zeros(B, K, 3, device=device)
    phi   = CanonicalFrame(R_w2c=R_w2c, t_w2c=t_w2c)

    # Identity world rotation accumulator
    R_obj_world = torch.eye(3, device=device).expand(B, K, 3, 3).contiguous()

    return SceneState(
        mu=mu_padded, cov=cov_padded, sh=sh_padded,
        opacity=opacity_padded, scale=scale_padded,
        phi=phi, mask=mask, R_obj_world=R_obj_world,
    )


# ══════════════════════════════════════════════════════════════════════
# Convert trajectory (list of SceneState) → GS4DSequence
# ══════════════════════════════════════════════════════════════════════

def _trajectory_to_gs4d(trajectory: List[SceneState]) -> GS4DSequence:
    """Stack a list of T SceneStates into our GS4DSequence format.

    SceneState has [B=1, K, N_max, ...] shape.  We collapse K and slice
    the real Gaussians (mask=True) → [T, N_real, ...].
    """
    T = len(trajectory)
    assert T > 0, "empty trajectory"

    # Determine a consistent N from the first scene's mask
    s0 = trajectory[0]
    mask0 = s0.mask[0]                                            # [K, N_max]
    valid_idx = mask0.flatten().nonzero(as_tuple=True)[0]         # [N_real]
    N_real = int(valid_idx.numel())
    if N_real == 0:
        # Fallback: use all positions (no mask filter)
        N_real = int(s0.mu.shape[1] * s0.mu.shape[2])

    mu_t      = np.zeros((T, N_real, 3),    dtype=np.float32)
    cov_t     = np.zeros((T, N_real, 3, 3), dtype=np.float32)
    sh0       = s0.sh[0].reshape(-1, s0.sh.shape[-1])             # [K*N_max, c_sh]
    opacity0  = s0.opacity[0].reshape(-1, 1)
    scale0    = s0.scale[0].reshape(-1, 3)

    sh_full      = sh0[valid_idx].cpu().numpy()                   # broadcast across T
    opacity_full = opacity0[valid_idx].cpu().numpy()
    scale_full   = scale0[valid_idx].cpu().numpy()

    for t, s in enumerate(trajectory):
        mu_flat  = s.mu[0].reshape(-1, 3)                         # [K*N_max, 3]
        cov_flat = s.cov[0].reshape(-1, 3, 3)
        if mu_flat.shape[0] != mask0.numel():
            # Shape changed mid-trajectory (unusual) — skip
            continue
        mu_t[t]  = mu_flat[valid_idx].detach().cpu().numpy()
        cov_t[t] = cov_flat[valid_idx].detach().cpu().numpy()

    sh_t      = np.broadcast_to(sh_full[None],      (T,) + sh_full.shape).copy()
    opacity_t = np.broadcast_to(opacity_full[None], (T,) + opacity_full.shape).copy()
    scale_t   = np.broadcast_to(scale_full[None],   (T,) + scale_full.shape).copy()

    return GS4DSequence(
        mu=mu_t, cov=cov_t, sh=sh_t, opacity=opacity_t, scale=scale_t,
    )


# ══════════════════════════════════════════════════════════════════════
# Public helpers (for diversity_eval.py and tests)
# ══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_action_tokens(
    model:         CAPModel,
    text:          str,
    num_samples:   int = 10,
    sampling_info: Optional[Dict[str, Any]] = None,
) -> List[List[int]]:
    """Sample N action token sequences from text — used for Diversity (PDF #9).

    Returns a list of N token id sequences.  Each sequence is a flat list of
    atomic-action token ids from Planner.sample_actions.
    """
    out = model.plan_from_text(
        texts=[text], sampling_info=sampling_info, num_samples=num_samples,
    )
    seqs_tensor = out.get("sequences")           # [num_samples, L]
    if seqs_tensor is None:
        return []
    return [seq.tolist() for seq in seqs_tensor]


@torch.no_grad()
def infer_one(
    model:        CAPModel,
    traj_dir:     Path,
    text:         str,
    cfg:          Dict,
    device:       torch.device,
    enable_physics: bool = True,
) -> Optional[GS4DSequence]:
    """Run text → 4DGS for one trajectory."""
    n_slots = int(cfg["encoder"]["object_encoder"]["slotatt_param"]["num_slots"])
    c_sh    = int(cfg["gs_param"]["gs_dimension"] - 11)

    scene = _build_initial_scene(traj_dir, n_gs_points=10000, c_sh=c_sh,
                                   n_slots=n_slots, device=device)
    out = model.infer_text(
        texts=[text], scene=scene,
        sampling_info=None, enable_physics=enable_physics,
    )
    trajectory = out.get("trajectory")
    if not trajectory:
        return None
    return _trajectory_to_gs4d(trajectory)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True, help="path to main_exp_final.pt")
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+",
                   default=["test_iid", "test_ood_unseen_pair",
                            "test_ood_unseen_object", "test_compositional_long"])
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--baseline-name", default="ours",
                   help="output bucket name; useful for multi-seed runs "
                        "(e.g. ours_s0, ours_s1, ours_s2 → aggregated to "
                        "ours mean ± std at format_latex time)")
    p.add_argument("--no-physics", action="store_true",
                   help="disable physics module during inference (debug)")
    p.add_argument("--limit", type=int, default=None,
                   help="for debugging, only process N trajectories per split")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip trajectories that already have pred_4dgs.npz")
    args = p.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ running on CPU — inference will be slow", file=sys.stderr)

    print(f"⏬ loading {args.ckpt}")
    model = load_model(args.ckpt, args.config, device)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n_total = n_ok = n_skip = n_fail = 0
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

            out_dir = baseline_output_dir(
                args.output_root, args.baseline_name,
                args.dataset_name, split, traj_id,
            )
            pred_path = out_dir / "pred_4dgs.npz"
            if args.skip_existing and pred_path.exists():
                n_skip += 1
                continue

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            try:
                seq = infer_one(model, traj_dir, text, cfg, device,
                                  enable_physics=not args.no_physics)
            except Exception as e:
                print(f"  ✗ {traj_id}  ERROR  {type(e).__name__}: {e}")
                n_fail += 1
                TrajMetrics(notes=f"ours_infer_failed: {e!r}").save(
                    out_dir / "metrics.json")
                continue

            if seq is None:
                n_fail += 1
                TrajMetrics(notes="ours_infer_empty_trajectory").save(
                    out_dir / "metrics.json")
                continue

            seq.save(pred_path)
            TrajMetrics(notes="pending_eval").save(out_dir / "metrics.json")
            n_ok += 1
            if n_ok <= 3 or n_ok % 50 == 0:
                print(f"  ✓ {traj_id}  T={seq.T} N={seq.N}")

    print(f"\n=== Ours inference complete ===")
    print(f"  total:        {n_total}")
    print(f"  ok:           {n_ok}")
    print(f"  skipped:      {n_skip} (already had pred_4dgs.npz)")
    print(f"  failed:       {n_fail}")
    print(f"  elapsed:      {time.time()-t0:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
