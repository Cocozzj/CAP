"""
k_scaling_sweep.py — Exp 3.1 (CAP/Experiment.md §3.C Tab 4, Fig 6).

Evaluate codebook-size scaling: load N checkpoints trained with different
K values (action_tokenizer.num_action_codebook), compute closure / inverse
gaps for each, and fit the theoretical decay  err ≈ A · K^(-1/d).

The exponent d acts as an *intrinsic dimension* of the action manifold —
fitting it from data confirms the codebook covers a low-dimensional set.

Run flow:
  1. ``--ckpts ckpt_K128.pt ckpt_K256.pt ...`` — pre-trained ckpts to evaluate.
     Their config (next to each ckpt) determines K — we read it directly,
     no need to specify K on the CLI.
  2. For each ckpt: load model+config, run ``algebraic_gaps`` style
     batched eval, average closure / inverse / commutator gaps.
  3. Fit ``log(err) = log(A) - (1/d) * log(K)`` via least-squares
     (a single line in log-log space).  Report fitted (A, d).

Output:
  - ``summary.json``   — per-K metrics + fitted exponents
  - ``points.csv``     — (K, closure, inverse, commutator) for plotting
  - ``fit.json``       — fitted A, d, R² for each metric

Usage::

    python -m eval.k_scaling_sweep \\
        --ckpts runs/ablK64/ckpt/main_exp_final.pt \\
                runs/ablK128/ckpt/main_exp_final.pt \\
                runs/ablK256/ckpt/main_exp_final.pt \\
                runs/ablK512/ckpt/main_exp_final.pt \\
                runs/ablK1024/ckpt/main_exp_final.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch

from model import build_scene_state
from model.loss import closure_loss, inverse_loss, commutator_loss
from dataloader import ToyDataset, collate_batch

from .utils import load_model_for_eval


# ──────────────────────────────────────────────────────────────────────
# Per-checkpoint eval
# ──────────────────────────────────────────────────────────────────────

def _gaps_for_ckpt(
    ckpt_path: Path, *,
    config_path: Path = None,
    device: str = "cuda",
    n_batches: int = 16,
    batch_size: int = 4,
    enable_physics: bool = False,
) -> Tuple[int, Dict[str, float]]:
    """Load a checkpoint and compute mean closure/inverse/commutator gaps.

    Returns (K, {closure, inverse, commutator}).
    """
    args = SimpleNamespace(
        ckpt   = str(ckpt_path),
        config = str(config_path) if config_path else None,
        device = device,
        output_dir = None,
    )
    model, cfg, dev = load_model_for_eval(args)
    K = int(cfg["encoder"]["action_tokenizer"]["num_action_codebook"])

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=n_batches * batch_size, sh_dim=sh_dim)

    clos, inv, comm = [], [], []
    with torch.no_grad():
        for b in range(n_batches):
            indices = list(range(b * batch_size, (b + 1) * batch_size))
            batch = collate_batch([ds[i] for i in indices])
            frames    = batch["frames"].to(dev)
            gs_params = [g.to(dev) for g in batch["gs_params"]]

            enc_out = model.encode(frames, gs_params=gs_params, tau=1.0)
            scene = build_scene_state(
                gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
            )
            ppseq = enc_out["physical_params"]

            clos.append(float(closure_loss(model.executor, scene, ppseq,
                                           enable_physics=enable_physics).item()))
            inv .append(float(inverse_loss(model.executor, scene, ppseq,
                                           enable_physics=enable_physics).item()))
            comm.append(float(commutator_loss(model.executor, scene, ppseq,
                                              enable_physics=enable_physics).item()))

    # Free GPU memory between checkpoints.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return K, {
        "closure":      float(np.mean(clos)),
        "closure_std":  float(np.std(clos)),
        "inverse":      float(np.mean(inv)),
        "inverse_std":  float(np.std(inv)),
        "commutator":   float(np.mean(comm)),
        "commutator_std": float(np.std(comm)),
    }


# ──────────────────────────────────────────────────────────────────────
# Power-law fit:  err ≈ A · K^(-1/d)
# ──────────────────────────────────────────────────────────────────────

def fit_power_law(Ks: List[int], errs: List[float]) -> Dict[str, float]:
    """Fit log(err) = log(A) - (1/d) log(K) via least squares.

    Returns {A, d, slope, intercept, r2, n}.
    Skips K's whose err is non-positive (log undefined).
    """
    pairs = [(K, e) for K, e in zip(Ks, errs)
             if K > 0 and e is not None and e > 0 and not math.isnan(e)]
    if len(pairs) < 2:
        return {"A": float("nan"), "d": float("nan"),
                "slope": float("nan"), "intercept": float("nan"),
                "r2": float("nan"), "n": len(pairs)}

    K_arr = np.asarray([p[0] for p in pairs], dtype=np.float64)
    e_arr = np.asarray([p[1] for p in pairs], dtype=np.float64)
    log_K = np.log(K_arr)
    log_e = np.log(e_arr)

    slope, intercept = np.polyfit(log_K, log_e, deg=1)        # log(e) = slope*log(K) + intercept
    pred = slope * log_K + intercept
    ss_res = float(((log_e - pred) ** 2).sum())
    ss_tot = float(((log_e - log_e.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    A = float(math.exp(intercept))
    d = float(-1.0 / slope) if slope != 0 else float("inf")

    return {
        "A":         A,
        "d":         d,
        "slope":     float(slope),
        "intercept": float(intercept),
        "r2":        float(r2),
        "n":         len(pairs),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True,
                   help="One or more stage-end checkpoints (different K each)")
    p.add_argument("--configs", nargs="*", default=None,
                   help="Optional explicit configs per ckpt (default: <ckpt>/../../config.yaml)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--n-batches", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--enable-physics", action="store_true")
    p.add_argument("--output-dir", type=str, default="runs/k_scaling_sweep")
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.configs is not None and len(args.configs) != len(args.ckpts):
        raise SystemExit("--configs (if given) must have same length as --ckpts")

    # ── Run per-K eval ──
    per_K: Dict[int, Dict[str, float]] = {}
    print(f"\n=== K-scaling sweep over {len(args.ckpts)} checkpoints ===\n")
    for i, ck in enumerate(args.ckpts):
        cfg_path = Path(args.configs[i]) if args.configs else None
        print(f"  [{i + 1}/{len(args.ckpts)}] {Path(ck).name}")
        K, gaps = _gaps_for_ckpt(
            Path(ck), config_path=cfg_path, device=args.device,
            n_batches=args.n_batches, batch_size=args.batch_size,
            enable_physics=args.enable_physics,
        )
        if K in per_K:
            print(f"    WARN: duplicate K={K}; overwriting earlier result")
        per_K[K] = gaps
        print(f"    K={K:>5d} | closure={gaps['closure']:.4f}  "
              f"inverse={gaps['inverse']:.4f}  comm={gaps['commutator']:.4f}")

    Ks_sorted = sorted(per_K.keys())

    # ── Fit power law per metric ──
    fits = {
        m: fit_power_law(Ks_sorted, [per_K[K][m] for K in Ks_sorted])
        for m in ("closure", "inverse", "commutator")
    }

    # ── Persist summary + CSV + fits ──
    summary = {
        "per_K":          {str(K): per_K[K] for K in Ks_sorted},
        "Ks":             Ks_sorted,
        "n_batches":      args.n_batches,
        "batch_size":     args.batch_size,
        "enable_physics": args.enable_physics,
        "fits":           fits,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")

    with open(out_dir / "points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "closure", "closure_std",
                    "inverse", "inverse_std",
                    "commutator", "commutator_std"])
        for K in Ks_sorted:
            r = per_K[K]
            w.writerow([K, r["closure"], r["closure_std"],
                        r["inverse"], r["inverse_std"],
                        r["commutator"], r["commutator_std"]])

    with open(out_dir / "fit.json", "w") as f:
        json.dump(fits, f, indent=2)

    # ── Pretty-print fits ──
    print("\n=== Power-law fits  (err ≈ A · K^(-1/d)) ===")
    for m, fit in fits.items():
        print(f"  {m:10s}: A={fit['A']:.4f}  d={fit['d']:.3f}  "
              f"R²={fit['r2']:.3f}  (n={fit['n']})")

    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
