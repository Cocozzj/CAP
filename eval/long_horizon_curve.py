"""
long_horizon_curve.py — Exp 1.2 helper (CAP/Experiment.md §3.A Tab 1, Fig 2).

Sweep generated sequence length T = 2..8 atomic steps and report the
text-conditioned success rate at each length.  This produces the
"success vs. plan length" curve that demonstrates *compounding* — a
trained model should degrade slower than the baseline as T grows.

For each (task, length T):
  1. Plan from text with ``max_len = T * K`` so the Planner is
     forced to emit at least T full timesteps before EOS.
  2. Truncate the unflattened plan to exactly T timesteps.
  3. Execute over an initial scene drawn from the eval dataset
     (n_trials trials per length).
  4. Test the per-task success criterion on the final state.

Output:
  - ``summary.json``    — { per_task: {task: {T: {success_rate, n_trials}}},
                            per_length: {T: success_rate_over_all_tasks} }
  - ``results.pt``      — torch dump of summary
  - ``curve.csv``       — long-format CSV ready for plotting

Usage::

    python -m eval.long_horizon_curve \\
        --ckpt runs/main_exp/seed_0/ckpt/main_exp_final.pt \\
        --tasks "open the drawer" "rotate the knob" "lift the cup" \\
        --lengths 2 3 4 5 6 7 8 \\
        --n-trials 8
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch

from model import build_scene_state
from dataload import collate_batch

from .utils import (add_common_eval_args, add_data_args, build_eval_loader,
                    get_output_dir, load_model_for_eval)
from .success_rate import SUCCESS_CRITERIA


# ──────────────────────────────────────────────────────────────────────
# Plan-at-length helper
# ──────────────────────────────────────────────────────────────────────

def _plan_for_length(model, text: str, K: int, T: int) -> torch.Tensor:
    """Sample a plan, force it to be exactly T timesteps long.

    The Planner stops at EOS by default, so we ask for ``max_len = T * K``
    and trim if it ran shorter (pad with last-valid token via
    ``model.unflatten_plan``).
    """
    plan_out = model.plan_from_text(
        texts=[text],
        sampling_info={"max_len": T * K, "num_samples": 1},
        num_samples=1,
    )
    plan_tokens = model.unflatten_plan(plan_out["sequences"], K=K)   # [1, T_eff, K]
    T_eff = plan_tokens.shape[1]
    if T_eff < T:
        # Plan was shorter than requested — repeat last frame to reach T.
        pad = plan_tokens[:, -1:, :].expand(-1, T - T_eff, -1).contiguous()
        plan_tokens = torch.cat([plan_tokens, pad], dim=1)
    elif T_eff > T:
        plan_tokens = plan_tokens[:, :T, :].contiguous()
    return plan_tokens                                               # [1, T, K]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_data_args(parser, default_split="test_compositional_long")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[2, 3, 4, 5, 6, 7, 8],
                        help="Sequence lengths (in atomic steps) to sweep.")
    parser.add_argument("--n-trials", type=int, default=8,
                        help="Initial scenes per (task, length) cell.")
    parser.add_argument("--enable-physics", action="store_true")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "long_horizon_curve")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds, loader = build_eval_loader(args, sh_dim, n_samples=args.n_trials)
    n_trials_actual = min(args.n_trials, len(ds))

    print(f"\n=== Long-horizon curve ===")
    print(f"  tasks   = {args.tasks}")
    print(f"  lengths = {args.lengths}")
    print(f"  trials  = {args.n_trials} per cell, physics={args.enable_physics}")

    per_task: Dict[str, Dict[int, Dict[str, float]]] = {}
    per_length_succ: Dict[int, List[int]] = {T: [] for T in args.lengths}

    with torch.no_grad():
        for txt in args.tasks:
            criterion = SUCCESS_CRITERIA.get(txt, SUCCESS_CRITERIA["default"])
            per_task[txt] = {}
            print(f"\n  task: {txt!r}")

            for T in args.lengths:
                successes = 0
                for trial in range(n_trials_actual):
                    batch = collate_batch([ds[trial]])
                    gs_params = [g.to(device) for g in batch["gs_params"]]
                    enc_out = model.encode(
                        batch["frames"].to(device), gs_params=gs_params, tau=1.0,
                    )
                    scene = build_scene_state(
                        gs_params=gs_params, phi=enc_out["phi"],
                        assignment=enc_out["assignment"],
                    )

                    plan_tokens = _plan_for_length(model, txt, K=scene.K, T=T)
                    ppseq = model.tokens_to_physical_params(plan_tokens)
                    exec_out = model.execute_sequence(
                        scene=scene, physical_params_seq=ppseq,
                        enable_physics=args.enable_physics,
                    )
                    ok = bool(criterion(scene, exec_out["final_state"]).any().item())
                    successes += int(ok)

                rate = successes / max(n_trials_actual, 1)
                per_task[txt][T] = {
                    "success_rate": rate,
                    "n_succeeded":  successes,
                    "n_trials":     n_trials_actual,
                }
                per_length_succ[T].append(rate)
                print(f"     T={T}: {rate * 100:5.1f}%  ({successes}/{n_trials_actual})")

    per_length = {
        T: float(sum(rs) / len(rs)) if rs else 0.0
        for T, rs in per_length_succ.items()
    }

    summary = {
        "per_task":       per_task,
        "per_length":     per_length,
        "lengths":        args.lengths,
        "tasks":          args.tasks,
        "n_trials":       args.n_trials,
        "enable_physics": args.enable_physics,
    }

    print("\n=== Mean success rate vs. length (avg over tasks) ===")
    for T in args.lengths:
        print(f"  T={T:2d}:  {per_length[T] * 100:5.1f}%")

    # JSON + torch dump
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")

    # Long-format CSV — one row per (task, T) for matplotlib/seaborn
    with open(out_dir / "curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "length", "success_rate", "n_succeeded", "n_trials"])
        for txt, by_T in per_task.items():
            for T, cell in by_T.items():
                w.writerow([txt, T, cell["success_rate"],
                            cell["n_succeeded"], cell["n_trials"]])
        for T, r in per_length.items():
            w.writerow(["__ALL__", T, r, "", args.n_trials * len(args.tasks)])

    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
