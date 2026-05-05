"""Physics counterfactual experiment (PDF f07d2c0a + fdfa011c 行 1101).

Cross-material / cross-mass / cross-friction generalisation:
For each material in cfg["executor"]["materials"], swap the model's runtime
physics params (via DeformSim.set_param_override) and re-execute the SAME
demonstration.  Compare the resulting trajectories to expose physics
generalisation behaviour.

Reports:
  - per-material trajectory end-state displacement
  - mass-vs-displacement curve (heavier objects should move LESS under same force)

Usage::
    python -m eval.physics_counterfactual \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import build_scene_state
from dataloader import ToyDataset, collate_batch

from .metrics import scene_distance_metric
from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--n-clips", type=int, default=4)
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "physics_counterfactual")

    materials = cfg.get("executor", {}).get("materials", {}) or model.materials
    if not materials:
        raise RuntimeError("No materials registered in cfg.executor.materials")
    print(f"\n=== Physics counterfactual over {len(materials)} materials ===")

    # Encode the same demo once
    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=args.n_clips, sh_dim=sh_dim)
    batch = collate_batch([ds[i] for i in range(args.n_clips)])
    gs_params = [g.to(device) for g in batch["gs_params"]]
    enc_out = model.encode(batch["frames"].to(device),
                                gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
    )
    physical_params_seq = enc_out["physical_params"]

    # Per-material execution
    results = {}
    for name, mat in materials.items():
        # Swap runtime physics params
        model.executor.deform.set_param_override(
            density        = mat["density"],
            youngs_modulus = mat["elastic_modulus"],
            poisson_ratio  = mat["poisson_ratio"],
            friction_coeff = mat["friction_coeff"],
        )

        with torch.no_grad():
            exec_out = model.execute_sequence(
                scene=scene,
                physical_params_seq=physical_params_seq,
                enable_physics=True,
            )

        d = scene_distance_metric(exec_out["final_state"], scene).item()
        results[name] = {
            "density":         float(mat["density"]),
            "elastic_modulus": float(mat["elastic_modulus"]),
            "friction_coeff":  float(mat["friction_coeff"]),
            "end_displacement": d,
        }
        print(f"  {name:8s}  ρ={mat['density']:7.1f}  μ={mat['friction_coeff']:.2f}  "
              f"→ end Δ={d:.4f}")

    # Always clear at the end so caller doesn't get stuck in override mode
    model.executor.deform.clear_param_override()

    # Save
    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(results, out_dir / "results.pt")
    print(f"  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
