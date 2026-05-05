"""Physics metrics (Physics Plugin PDF §2.2 explicit eval list).

  - 体积保持率  (volume_preservation)        : |V_after / V_before − 1|
  - 弹性恢复率  (elastic_recovery)           : after T steps with no force,
                                              how close to rest pose?
  - 接触/穿透罚 (penetration_penalty)        : count of Gaussians with z < 0
  - 能量稳定性  (energy_stability)           : KE+PE drift from step 0 to T

These complement the cross-material counterfactual already in
``physics_counterfactual.py`` — together they cover the PDF's full
"physics plugin evaluation" section.

Usage::
    python -m eval.physics_metrics --ckpt runs/main_exp/ckpt/main_exp_final.pt
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch

from model import build_scene_state, SceneState
from model.utils import masked_mean

from .utils import (add_common_eval_args, add_data_args, build_eval_loader,
                    get_output_dir, load_model_for_eval)


# ──────────────────────────────────────────────────────────────────────
# Per-trajectory physics summary
# ──────────────────────────────────────────────────────────────────────

def _per_object_volume(state: SceneState) -> torch.Tensor:
    """Estimate per-object volume as bbox extent product (mask-aware).

    Returns: [B, K]
    """
    if state.mask is None:
        max_p = state.mu.max(dim=2).values
        min_p = state.mu.min(dim=2).values
    else:
        # Replace masked-out positions with COM so they don't expand bbox
        com = masked_mean(state.mu, state.mask, dim=2, keepdim=True)
        m = state.mask.unsqueeze(-1).float()
        pos = state.mu * m + com * (1 - m)
        max_p = pos.max(dim=2).values
        min_p = pos.min(dim=2).values
    extent = (max_p - min_p).clamp(min=1e-6)
    return extent.prod(dim=-1)                             # [B, K]


def volume_preservation(initial: SceneState, final: SceneState) -> float:
    """Mean |V_after / V_before − 1| across all objects + batch."""
    v_i = _per_object_volume(initial)
    v_f = _per_object_volume(final)
    ratio = v_f / v_i.clamp(min=1e-6)
    return float((ratio - 1.0).abs().mean())


def elastic_recovery(rest: SceneState, perturbed: SceneState) -> float:
    """Per-Gaussian L2 distance from rest pose (mask-aware)."""
    diff = (rest.mu - perturbed.mu).norm(dim=-1)
    if rest.mask is not None:
        return float(masked_mean(diff, rest.mask, dim=-1).mean())
    return float(diff.mean())


def penetration_penalty(state: SceneState, ground_z: float = 0.0) -> float:
    """Mean penetration depth (positive = penetrated)."""
    pen = (ground_z - state.mu[..., 2]).clamp(min=0)         # [B, K, N]
    if state.mask is not None:
        return float(masked_mean(pen, state.mask, dim=-1).mean())
    return float(pen.mean())


def energy_stability(trajectory: List[SceneState],
                     gravity: float = 9.81) -> float:
    """Std of (KE + PE) across timesteps, normalised by mean.  Lower = more stable.

    KE proxy: per-object COM displacement squared (no velocity available).
    PE proxy: per-object COM z-coordinate × gravity.
    """
    if len(trajectory) < 2:
        return 0.0

    energies = []
    prev_com = None
    for s in trajectory:
        if s.mask is None:
            com = s.mu.mean(dim=2)                     # [B, K, 3]
        else:
            com = masked_mean(s.mu, s.mask, dim=2)
        pe = (gravity * com[..., 2]).mean()             # scalar
        if prev_com is not None:
            ke = ((com - prev_com).norm(dim=-1) ** 2).mean()
        else:
            ke = torch.zeros(())
        energies.append((ke + pe).item())
        prev_com = com

    e = torch.tensor(energies)
    mean_e = e.mean().abs().clamp(min=1e-6)
    return float(e.std(unbiased=False) / mean_e)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_data_args(parser, default_split="test_iid")
    parser.add_argument("--n-batches",  type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ground-z",   type=float, default=0.0,
                        help="Match cfg.executor.physics.rigid_contact.ground_height")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "physics_metrics")

    print(f"\n=== Physics metrics: {args.n_batches} batches × {args.batch_size} samples ===")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    dataset, loader = build_eval_loader(
        args, sh_dim,
        n_samples=args.n_batches * args.batch_size,
        batch_size=args.batch_size,
    )
    n_batches_actual = min(args.n_batches, len(loader))

    vols, recs, pens, ens = [], [], [], []

    with torch.no_grad():
        for b, batch in enumerate(loader):
            if b >= n_batches_actual:
                break
            frames    = batch["frames"].to(device)
            gs_params = [g.to(device) for g in batch["gs_params"]]

            enc_out = model.encode(frames, gs_params=gs_params, tau=1.0)
            scene = build_scene_state(
                gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
            )
            ppseq = enc_out["physical_params"]

            exec_out = model.execute_sequence(
                scene=scene, physical_params_seq=ppseq, enable_physics=True,
            )

            vols.append(volume_preservation(scene, exec_out["final_state"]))
            recs.append(elastic_recovery(scene, exec_out["final_state"]))
            pens.append(penetration_penalty(exec_out["final_state"], args.ground_z))
            ens.append(energy_stability(exec_out["trajectory"]))

            print(f"  batch {b + 1:>3d}/{n_batches_actual}: "
                  f"vol={vols[-1]:.4f}  recov={recs[-1]:.4f}  "
                  f"penet={pens[-1]:.4f}  energy={ens[-1]:.4f}")

    summary = {
        "volume_preservation":  {"mean": float(torch.tensor(vols).mean()),
                                  "std":  float(torch.tensor(vols).std(unbiased=False))},
        "elastic_recovery":     {"mean": float(torch.tensor(recs).mean()),
                                  "std":  float(torch.tensor(recs).std(unbiased=False))},
        "penetration_penalty":  {"mean": float(torch.tensor(pens).mean()),
                                  "std":  float(torch.tensor(pens).std(unbiased=False))},
        "energy_stability":     {"mean": float(torch.tensor(ens).mean()),
                                  "std":  float(torch.tensor(ens).std(unbiased=False))},
        "n_batches": args.n_batches,
        "batch_size": args.batch_size,
    }
    print("\n=== Summary ===")
    for name in ("volume_preservation", "elastic_recovery",
                 "penetration_penalty", "energy_stability"):
        s = summary[name]
        print(f"  {name:24s}  {s['mean']:.4f} ± {s['std']:.4f}")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")
    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
