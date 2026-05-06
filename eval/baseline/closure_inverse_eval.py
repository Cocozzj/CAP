"""Closure / Inverse Gap evaluation (PDF §3 algebraic structure).

For our hierarchical action-token model, two algebraic properties of the
learned codebook are evaluated per trajectory:

  Closure   : d_M( E(g_b) ∘ E(g_a) ,  E( g_a ⊙̂ g_b ) )
              — applying tokens a then b should equal applying the
                algebraically-composed token (a ⊙̂ b) once.
  Inverse   : d_M( E(g) ∘ E(ĝ⁻¹) ,  id )
              — applying a token then its learned inverse should be the
                identity.

Both are computed by the same code that drives the training loss
(``model.loss.closure_loss`` / ``inverse_loss``).  We just (1) sample
plan tokens from the model for each test text, (2) decode them to
physical_params_seq, (3) call the loss helpers, and (4) write the
scalars into the existing ``metrics.json``.

Usage:

    python -m eval.baseline.closure_inverse_eval \\
        --ckpt   runs/main_a/seed_0/ckpt/main_exp_final.pt \\
        --config configs/config.yaml \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines \\
        --baseline-name ours_s0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import torch

from dataload.text import task_to_text
from model.loss import closure_loss, inverse_loss

from .common import TrajMetrics, baseline_output_dir, iter_split_entries
from .ours.runner import _build_initial_scene, load_model


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",     required=True)
    p.add_argument("--config",   default="configs/config.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits",   nargs="+", default=["test_iid"])
    p.add_argument("--output-root",   default="runs/baselines")
    p.add_argument("--dataset-name",  default="dataset_a")
    p.add_argument("--baseline-name", default="ours_s0",
                   help="must match the directory the ours runner wrote to "
                        "(e.g. ours_s0 / ours_s1 / ours_s2)")
    p.add_argument("--n-samples-per-traj", type=int, default=4,
                   help="closure/inverse are stochastic (random t in plan); "
                        "average over N samples for a stable per-traj number")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true",
                   help="skip trajs that already have closure_gap filled")
    args = p.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("⚠ no GPU — closure/inverse will be slow", file=sys.stderr)

    print(f"⏬ loading {args.ckpt}")
    model = load_model(args.ckpt, args.config, device)
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    n_slots = int(cfg["encoder"]["object_encoder"]["slotatt_param"]["num_slots"])
    c_sh    = int(cfg["gs_param"]["gs_dimension"] - 11)

    n_total = n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        for traj_id, traj_dir, entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            if args.limit is not None and n_total >= args.limit:
                break
            n_total += 1

            out_dir = baseline_output_dir(
                args.output_root, args.baseline_name,
                args.dataset_name, split, traj_id,
            )
            metrics_path = out_dir / "metrics.json"
            if not metrics_path.exists():
                n_fail += 1
                continue
            try:
                m = TrajMetrics.load(metrics_path)
            except Exception:
                n_fail += 1
                continue
            if args.skip_existing and m.closure_gap is not None and m.inverse_gap is not None:
                n_skip += 1
                continue

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            try:
                with torch.no_grad():
                    scene = _build_initial_scene(
                        Path(traj_dir), n_gs_points=10000, c_sh=c_sh,
                        n_slots=n_slots, device=device,
                    )
                    plan_out = model.plan_from_text(
                        texts=[text], sampling_info=None, num_samples=1,
                    )
                    K = scene.K
                    plan_tokens = model.unflatten_plan(
                        plan_out["sequences"], K=K,
                    )
                    pp_seq = model.tokens_to_physical_params(plan_tokens)
                    # tokens_to_physical_params returns per-step tensors of
                    # shape [B, T, K, ...] (T = plan T_macro).  closure_loss
                    # requires T ≥ 2; if planner emits a single macro-step,
                    # we duplicate it (the closure check then degenerates to
                    # 0 — i.e. a single token trivially commutes with itself).

                    # Average closure / inverse over N samples for stability
                    cls_vals, inv_vals = [], []
                    for _ in range(args.n_samples_per_traj):
                        cls = closure_loss(model.executor, scene, pp_seq,
                                           enable_physics=False)
                        inv = inverse_loss(model.executor, scene, pp_seq,
                                           enable_physics=False)
                        cls_vals.append(float(cls.item()))
                        inv_vals.append(float(inv.item()))

                m.closure_gap = float(sum(cls_vals) / len(cls_vals))
                m.inverse_gap = float(sum(inv_vals) / len(inv_vals))
                m.save(metrics_path)
                n_ok += 1
                if n_ok <= 3 or n_ok % 100 == 0:
                    print(f"  ✓ {traj_id}  "
                          f"closure={m.closure_gap:.4f}  inverse={m.inverse_gap:.4f}")
            except Exception as e:
                n_fail += 1
                if n_fail <= 3:
                    print(f"  ✗ {traj_id}  {type(e).__name__}: {e}")

    print(f"\n=== Closure/Inverse complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skip}")
    print(f"  failed:  {n_fail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
