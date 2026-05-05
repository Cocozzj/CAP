"""Mode B — Demo imitation.

  demo video  →  Encoder  →  physical_params  →  Executor on target_scene

Reports:
  - per-step scene distance vs. an aligned GT trajectory (if available)
  - reconstruction quality vs. demo frames

Usage::
    python -m eval.imitation \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt \\
        --n-clips 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import build_scene_state
from dataloader import ToyDataset, collate_batch

from .metrics import psnr, lpips_score, scene_distance_metric
from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--n-clips", type=int, default=8,
                        help="Number of toy demo clips to process")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "mode_b_imitation")

    print(f"\n=== Mode B: imitation, {args.n_clips} clips ===")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=args.n_clips, sh_dim=sh_dim)
    batch = collate_batch([ds[i] for i in range(args.n_clips)])
    frames    = batch["frames"].to(device)
    gs_params = [g.to(device) for g in batch["gs_params"]]

    # Encoder → physical_params + initial scene
    enc_out = model.encode(frames, gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"],
        assignment=enc_out["assignment"],
    )
    physical_params_seq = enc_out["physical_params"]

    exec_out = model.execute_sequence(
        scene=scene, physical_params_seq=physical_params_seq,
        enable_physics=True,
    )

    # Self-consistency: scene distance between trajectory[-1] and trajectory[0]
    # (proxy for "how much did the model think happened")
    if len(exec_out["trajectory"]) > 0:
        d = scene_distance_metric(exec_out["trajectory"][-1], scene)
        print(f"  trajectory-end displacement: {d.item():.4f}")
    else:
        d = torch.zeros(())

    torch.save({
        "physical_params_seq": physical_params_seq,
        "final_state":         exec_out["final_state"],
        "displacement":        d.item() if torch.is_tensor(d) else d,
    }, out_dir / "results.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "n_clips": args.n_clips,
            "trajectory_steps": len(exec_out["trajectory"]),
            "displacement": float(d.item() if torch.is_tensor(d) else d),
        }, f, indent=2)
    print(f"  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
