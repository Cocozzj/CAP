"""Task success rate (PDF main proposal lines 605, 1086).

  text → action → scene → check if task-specific success criterion is met

The success criteria are PROJECT-SPECIFIC and need to be defined per task.
This module provides a small framework + a registry of stub criteria for
the toy data; replace ``SUCCESS_CRITERIA`` with your real task definitions.

Two evaluations:
  1. Standard: text → action → success?
  2. Zero-shot composition: train on (door-open, drawer-pull),
     test on UNSEEN combination (drawer-open) — set --zero-shot to enable.

Usage::
    python -m eval.success_rate \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt \\
        --tasks "open the drawer" "rotate the knob"
"""

from __future__ import annotations

import argparse
import json
from typing import Callable, Dict, List

import torch

from model import build_scene_state, SceneState
from model.utils import masked_mean
from dataload import collate_batch

from .utils import (add_common_eval_args, add_data_args, build_eval_loader,
                    get_output_dir, load_model_for_eval)


# ──────────────────────────────────────────────────────────────────────
# Task success criteria — replace with your actual task definitions
# ──────────────────────────────────────────────────────────────────────
# Each criterion takes (initial_state, final_state) and returns a bool
# Tensor of shape [B] (one success flag per batch element).
#
# Examples below are SCAFFOLDING ONLY — they use generic criteria
# (e.g., "object centroid moved more than threshold").  Replace with
# task-specific physics: door angle, drawer pull distance, etc.

def _com_displacement(initial: SceneState, final: SceneState) -> torch.Tensor:
    """Per-object centroid displacement [B, K]."""
    if initial.mask is None:
        m_i = initial.mu.mean(dim=2)
        m_f = final.mu.mean(dim=2)
    else:
        m_i = masked_mean(initial.mu, initial.mask, dim=2)
        m_f = masked_mean(final.mu,   final.mask,   dim=2)
    return (m_f - m_i).norm(dim=-1)            # [B, K]


def crit_object_moved(initial: SceneState, final: SceneState,
                      threshold: float = 0.05, slot: int = 0) -> torch.Tensor:
    """Generic: object at slot ``slot`` moved more than ``threshold`` metres."""
    return _com_displacement(initial, final)[:, slot] > threshold


def crit_object_lifted(initial: SceneState, final: SceneState,
                       threshold: float = 0.02, slot: int = 0) -> torch.Tensor:
    """Generic: object at slot ``slot`` raised by more than ``threshold`` along z."""
    if initial.mask is None:
        m_i = initial.mu[..., 2].mean(dim=2)
        m_f = final.mu[..., 2].mean(dim=2)
    else:
        m_i = masked_mean(initial.mu[..., 2:3], initial.mask, dim=2).squeeze(-1)
        m_f = masked_mean(final.mu[..., 2:3],   final.mask,   dim=2).squeeze(-1)
    return (m_f[:, slot] - m_i[:, slot]) > threshold


# Register tasks here.  In a real project, key by task name from your dataset.
SUCCESS_CRITERIA: Dict[str, Callable[..., torch.Tensor]] = {
    "open the drawer":   crit_object_moved,
    "close the lid":     crit_object_moved,
    "rotate the knob":   crit_object_moved,
    "push the button":   crit_object_moved,
    "lift the cup":      crit_object_lifted,
    "pour the bottle":   crit_object_moved,
    "press the lever":   crit_object_moved,
    "fold the cloth":    crit_object_moved,
    "default":           crit_object_moved,
}


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_data_args(parser, default_split="test_iid")
    parser.add_argument("--tasks",      nargs="+", required=True)
    parser.add_argument("--n-trials",   type=int, default=8,
                        help="Number of trial scenes per task")
    parser.add_argument("--zero-shot",  action="store_true",
                        help="Filter out tasks present in TRAIN_TASKS list")
    parser.add_argument("--enable-physics", action="store_true")
    args = parser.parse_args()

    # Set TRAIN_TASKS to your actual training tasks if using --zero-shot
    TRAIN_TASKS = {"open the drawer", "rotate the knob"}     # placeholder

    if args.zero_shot:
        unseen = [t for t in args.tasks if t not in TRAIN_TASKS]
        if not unseen:
            print(f"WARNING: all requested tasks are in TRAIN_TASKS — zero-shot test trivial.")
        eval_tasks = unseen or args.tasks
    else:
        eval_tasks = args.tasks

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "success_rate")

    print(f"\n=== Success rate eval ===")
    print(f"  zero_shot={args.zero_shot}, n_trials={args.n_trials}, "
          f"enable_physics={args.enable_physics}")

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds, loader = build_eval_loader(args, sh_dim, n_samples=args.n_trials)
    n_trials_actual = min(args.n_trials, len(ds))

    per_task: Dict[str, dict] = {}

    for txt in eval_tasks:
        print(f"\n  task: {txt!r}")
        criterion = SUCCESS_CRITERIA.get(txt, SUCCESS_CRITERIA["default"])

        # Plan the action from text (single sample for determinism)
        plan_out = model.plan_from_text(texts=[txt], num_samples=1)
        sequences = plan_out["sequences"]                       # [1, L_out]

        successes = []
        with torch.no_grad():
            for trial in range(n_trials_actual):
                batch = collate_batch([ds[trial]])
                gs_params = [g.to(device) for g in batch["gs_params"]]
                enc_out = model.encode(batch["frames"].to(device),
                                            gs_params=gs_params, tau=1.0)
                scene = build_scene_state(
                    gs_params=gs_params, phi=enc_out["phi"],
                    assignment=enc_out["assignment"],
                )

                K = scene.K
                plan_tokens = model.unflatten_plan(sequences, K=K)
                ppseq = model.tokens_to_physical_params(plan_tokens)
                exec_out = model.execute_sequence(
                    scene=scene, physical_params_seq=ppseq,
                    enable_physics=args.enable_physics,
                )
                ok = criterion(scene, exec_out["final_state"]).bool()
                successes.append(int(ok.any().item()))

        rate = sum(successes) / len(successes) if successes else 0.0
        per_task[txt] = {
            "success_rate":   rate,
            "trials":         len(successes),
            "n_succeeded":    sum(successes),
            "criterion_name": criterion.__name__,
        }
        print(f"     success rate = {rate * 100:.1f}%  "
              f"({sum(successes)}/{len(successes)})")

    overall = (sum(d["n_succeeded"] for d in per_task.values())
               / max(sum(d["trials"] for d in per_task.values()), 1))

    summary = {
        "per_task": per_task,
        "overall_success_rate": overall,
        "zero_shot": args.zero_shot,
        "enable_physics": args.enable_physics,
    }
    print(f"\n=== Overall success rate: {overall * 100:.1f}% ===")

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")
    print(f"  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
