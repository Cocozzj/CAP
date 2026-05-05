"""Trajectory-level metrics (PDF main proposal lines 603, 1076).

Treats per-object COM (or per-part centre, if part labels available) as
"joints" and computes:

  ADE   — Average Displacement Error      (mean over T of ||pred − gt||)
  FDE   — Final Displacement Error        (||pred[-1] − gt[-1]||)
  MPJPE — Mean Per-Joint Position Error   (mean over joints of ADE)

For the toy dataset (no GT trajectories), this script self-pairs by
running the model twice with different random seeds and reporting the
"reproducibility" version of these metrics — that's a useful sanity
check, but for real Dataset-A you should pair with the simulator GT
trajectories instead.

Usage::
    python -m eval.trajectory_metrics --ckpt runs/main_exp/ckpt/main_exp_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch

from model import build_scene_state, SceneState
from model.utils import masked_mean
from dataloader import ToyDataset, collate_batch

from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


def _per_object_com(state: SceneState) -> torch.Tensor:
    """Compute mask-aware per-object COM.  Returns [B, K, 3]."""
    if state.mask is None:
        return state.mu.mean(dim=2)
    return masked_mean(state.mu, state.mask, dim=2, keepdim=False)


def trajectory_ADE_FDE_MPJPE(
    pred_traj: List[SceneState],
    gt_traj:   List[SceneState],
) -> dict:
    """Compute (ADE, FDE, MPJPE) over a paired trajectory.

    Conventions:
      - "joints" = K objects (per-object COM tracking)
      - ADE is over time steps (averaged across all joints)
      - FDE is at final step only
      - MPJPE is per-joint then averaged
    """
    assert len(pred_traj) == len(gt_traj), "trajectory lengths differ"
    if not pred_traj:
        return {"ADE": float("nan"), "FDE": float("nan"), "MPJPE": float("nan")}

    # [T, B, K, 3]
    pred_coms = torch.stack([_per_object_com(s) for s in pred_traj], dim=0)
    gt_coms   = torch.stack([_per_object_com(s) for s in gt_traj],   dim=0)
    diff = (pred_coms - gt_coms).norm(dim=-1)                        # [T, B, K]

    ade = diff.mean()                                                # mean over T,B,K
    fde = diff[-1].mean()                                            # last step
    mpjpe = diff.mean(dim=0).mean()                                  # per-joint avg
    return {"ADE": float(ade), "FDE": float(fde), "MPJPE": float(mpjpe)}


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--n-batches",  type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--enable-physics", action="store_true")
    parser.add_argument("--mode", type=str, default="self-pair",
                        choices=["self-pair"],
                        help="self-pair: run model twice with different seeds (toy use). "
                             "TODO: real-gt mode once Dataset-A loader is plugged in.")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "trajectory_metrics")

    print(f"\n=== Trajectory metrics ({args.mode}, n_batches={args.n_batches}) ===")
    print(f"  enable_physics = {args.enable_physics}")
    if args.mode == "self-pair":
        print("  NOTE: using self-pair (model run twice with different seeds).")
        print("        For real evaluation, pair against PartNet-Mobility GT trajectories.")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    dataset = ToyDataset(n_samples=args.n_batches * args.batch_size, sh_dim=sh_dim)

    ade_vals, fde_vals, mpjpe_vals = [], [], []

    with torch.no_grad():
        for b in range(args.n_batches):
            indices = list(range(b * args.batch_size, (b + 1) * args.batch_size))
            batch = collate_batch([dataset[i] for i in indices])
            frames    = batch["frames"].to(device)
            gs_params = [g.to(device) for g in batch["gs_params"]]

            enc_out = model.encode(frames, gs_params=gs_params, tau=1.0)
            scene = build_scene_state(
                gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
            )
            ppseq = enc_out["physical_params"]

            # Run #1
            torch.manual_seed(b * 2)
            out_a = model.execute_sequence(scene=scene, physical_params_seq=ppseq,
                                          enable_physics=args.enable_physics)
            # Run #2 (different seed → router stochasticity / sampling)
            torch.manual_seed(b * 2 + 1)
            out_b = model.execute_sequence(scene=scene, physical_params_seq=ppseq,
                                          enable_physics=args.enable_physics)

            metrics = trajectory_ADE_FDE_MPJPE(out_a["trajectory"], out_b["trajectory"])
            ade_vals.append(metrics["ADE"])
            fde_vals.append(metrics["FDE"])
            mpjpe_vals.append(metrics["MPJPE"])
            print(f"  batch {b + 1:>3d}/{args.n_batches}: "
                  f"ADE={metrics['ADE']:.4f}  FDE={metrics['FDE']:.4f}  "
                  f"MPJPE={metrics['MPJPE']:.4f}")

    summary = {
        "ADE":   {"mean": float(torch.tensor(ade_vals).mean()),
                  "std":  float(torch.tensor(ade_vals).std(unbiased=False))},
        "FDE":   {"mean": float(torch.tensor(fde_vals).mean()),
                  "std":  float(torch.tensor(fde_vals).std(unbiased=False))},
        "MPJPE": {"mean": float(torch.tensor(mpjpe_vals).mean()),
                  "std":  float(torch.tensor(mpjpe_vals).std(unbiased=False))},
        "mode":  args.mode,
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
    }
    print("\n=== Summary (units: metres) ===")
    for name in ("ADE", "FDE", "MPJPE"):
        s = summary[name]
        print(f"  {name:6s}  {s['mean']:.4f} ± {s['std']:.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")
    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
