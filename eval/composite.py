"""Mode C — Composite task (PDF §4.1 Prop 4).

  texts = ["pick up cup", "place on shelf"]
                       │
                       ▼
   for each text:  plan_from_text → seq_i
                       │
                       ▼
   full_seq = concat(seq_1, seq_2, ...)
                       │
                       ▼
   Executor.apply_sequence

Reports the composite-task generated sequence length and (optionally) a
success/failure flag per task definition.

Usage::
    python -m eval.composite \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt \\
        --tasks "pick up the cup" "place it on the shelf"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import build_scene_state
from dataloader import ToyDataset, collate_batch

from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--tasks",      nargs="+", required=True,
                        help="Two or more sub-task text prompts")
    parser.add_argument("--n-scenes",   type=int, default=4)
    args = parser.parse_args()
    assert len(args.tasks) >= 2, "Need at least 2 sub-tasks for composite mode"

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "mode_c_composite")

    print(f"\n=== Mode C: composite, {len(args.tasks)} sub-tasks ===")

    # First, get task embeddings for each sub-task
    task_embs = []
    for txt in args.tasks:
        info = model.text_to_task([txt])
        task_embs.append(info["task_emb"])         # [1, task_dim]
        print(f"  '{txt}' → task_id {info.get('task_id', '?')}")

    # Concatenate via plan_composite_from_texts
    plan_out = model.plan_composite_from_texts(args.tasks)
    full_seq = plan_out["full_seq"]
    sub_seqs = plan_out["sub_seqs"]

    print(f"  sub-sequence lengths: {[s.shape[1] for s in sub_seqs]}")
    print(f"  concatenated length: {full_seq.shape[1]}")

    # Execute on a toy scene
    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=args.n_scenes, sh_dim=sh_dim)
    batch = collate_batch([ds[i] for i in range(args.n_scenes)])
    gs_params = [g.to(device) for g in batch["gs_params"]]
    enc_out = model.encode(batch["frames"].to(device),
                                gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
    )

    K = scene.K
    plan_tokens = model.unflatten_plan(full_seq[:args.n_scenes], K=K)
    physical_params_seq = model.tokens_to_physical_params(plan_tokens)
    exec_out = model.execute_sequence(
        scene=scene, physical_params_seq=physical_params_seq,
        enable_physics=True,
    )

    torch.save({
        "tasks": args.tasks, "full_seq": full_seq, "sub_seqs": sub_seqs,
        "trajectory_steps": len(exec_out["trajectory"]),
    }, out_dir / "results.pt")
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "tasks": args.tasks,
            "sub_lens": [int(s.shape[1]) for s in sub_seqs],
            "total_len": int(full_seq.shape[1]),
        }, f, indent=2)
    print(f"  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
