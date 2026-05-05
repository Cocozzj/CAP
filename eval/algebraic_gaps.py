"""Algebraic-structure gaps as eval metrics (PDF main proposal lines 599-624).

Forward-only versions of the loss-suite's closure / inverse / commutator
losses, aggregated over many batches with mean ± std.  These three numbers
are the *primary* algebraic metrics the PDF requires for the main results
table — separate from training loss values because:

  - eval is computed under ``model.eval()`` and ``torch.no_grad()``
  - aggregated over a large held-out split, not a single batch
  - reported in scene-distance units (metres), not loss-scaled

Usage::
    python -m eval.algebraic_gaps --ckpt runs/main_exp/ckpt/main_exp_final.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import torch

from model import build_scene_state
from model.loss import closure_loss, inverse_loss, commutator_loss
from dataloader import ToyDataset, collate_batch

from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


def _aggregate(values: List[float]) -> dict:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    t = torch.tensor(values)
    return {
        "mean": float(t.mean()),
        "std":  float(t.std(unbiased=False)),
        "n":    len(values),
    }


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--n-batches",  type=int, default=16,
                        help="Number of held-out batches to aggregate over")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--enable-physics", action="store_true",
                        help="Run gaps in physics-ON mode (Stage 1/2 eval)")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "algebraic_gaps")

    print(f"\n=== Algebraic gaps: {args.n_batches} batches × {args.batch_size} samples ===")
    print(f"  enable_physics = {args.enable_physics}")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    dataset = ToyDataset(n_samples=args.n_batches * args.batch_size, sh_dim=sh_dim)

    clos_vals, inv_vals, comm_vals = [], [], []

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

            clos = closure_loss(model.executor, scene, ppseq,
                                enable_physics=args.enable_physics)
            inv  = inverse_loss(model.executor, scene, ppseq,
                                enable_physics=args.enable_physics)
            comm = commutator_loss(model.executor, scene, ppseq,
                                   enable_physics=args.enable_physics)

            clos_vals.append(float(clos.item()))
            inv_vals.append(float(inv.item()))
            comm_vals.append(float(comm.item()))

            print(f"  batch {b + 1:>3d}/{args.n_batches}: "
                  f"clos={clos_vals[-1]:.4f}  inv={inv_vals[-1]:.4f}  "
                  f"comm={comm_vals[-1]:.4f}")

    summary = {
        "closure_gap":      _aggregate(clos_vals),
        "inverse_gap":      _aggregate(inv_vals),
        "commutator_dev":   _aggregate(comm_vals),
        "enable_physics":   args.enable_physics,
        "n_batches":        args.n_batches,
        "batch_size":       args.batch_size,
    }
    print("\n=== Summary (units: scene-distance, metres) ===")
    for name in ("closure_gap", "inverse_gap", "commutator_dev"):
        s = summary[name]
        print(f"  {name:18s}  {s['mean']:.4f} ± {s['std']:.4f}  (n={s['n']})")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")
    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
