"""Mode A — Text-conditioned generation.

    text  →  Planner.sample_actions  →  atomic seq  →  Executor.apply_sequence
            → reconstructed trajectory

Reports:
  - generated atomic-token sequences (saved to .pt)
  - codebook utilisation stats
  - if --gt provided: PSNR/LPIPS vs GT frames

Usage::
    python -m eval.text_conditioned \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt \\
        --texts "open the drawer" "rotate the knob"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import codebook_utilisation, build_scene_state

from .utils import (add_common_eval_args, add_data_args, build_eval_loader,
                    get_output_dir, load_model_for_eval)
from .render_hook import render_scene, available_backend
from .metrics import psnr, lpips_score


def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    add_data_args(parser, default_split="test_iid")
    parser.add_argument("--texts",       nargs="+", required=True,
                        help="One or more text prompts")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--n-scenes",    type=int, default=4,
                        help="Number of scenes to sample for execution")
    parser.add_argument("--render", action="store_true",
                        help="Try rendering final SceneState via gsplat/nerfacc + "
                             "compute PSNR/LPIPS vs GT frames (skipped if no renderer)")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "mode_a_text")

    print(f"\n=== Mode A: text-conditioned, {len(args.texts)} prompts ===")

    # 1) Generate atomic sequences from text
    plan_out = model.plan_from_text(
        texts=args.texts, num_samples=args.num_samples,
    )
    sequences = plan_out["sequences"]        # [B*N, L_out]
    print(f"  generated sequences shape: {tuple(sequences.shape)}")

    # 2) Codebook utilisation
    util = codebook_utilisation(
        sequences, num_codes=model.encoder.action_enc.vq.num_codes,
    )
    print(f"  codebook utilisation: {util}")

    # 3) Execute on a real scene + read out trajectory length
    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds, loader = build_eval_loader(
        args, sh_dim, n_samples=args.n_scenes, batch_size=args.n_scenes,
    )
    batch = next(iter(loader))
    gs_params = [g.to(device) for g in batch["gs_params"]]

    # Build a SceneState with the encoder's phi
    enc_out = model.encode(batch["frames"].to(device), gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"],
        assignment=enc_out["assignment"],
    )

    K = scene.K
    plan_tokens = model.unflatten_plan(sequences[:args.n_scenes], K=K)
    physical_params_seq = model.tokens_to_physical_params(plan_tokens)

    exec_out = model.execute_sequence(
        scene=scene, physical_params_seq=physical_params_seq,
        enable_physics=True,
    )
    print(f"  trajectory length: {len(exec_out['trajectory'])} steps")

    # 4) Optional: render → PSNR/LPIPS vs GT frames
    image_metrics = None
    if args.render:
        backend = available_backend()
        print(f"\n  render backend: {backend!r}")
        if backend is None:
            print("  → no renderer installed; skipping image metrics.")
        else:
            # Use the input toy frames as a stand-in for GT (real loader should
            # supply per-camera GT frames matched to the trajectory).
            rendered = render_scene(exec_out["final_state"],
                                    camera_params={}, image_size=(64, 64))
            if rendered is not None:
                # Compare last-frame across views
                gt_last = batch["frames"][:, :, -1]      # [B, V, 3, H, W]
                psnr_v  = psnr(rendered, gt_last)
                lpips_v = lpips_score(rendered, gt_last)
                image_metrics = {"PSNR": float(psnr_v), "LPIPS": float(lpips_v)}
                print(f"  PSNR = {psnr_v:.2f}  LPIPS = {lpips_v:.4f}")
            else:
                print("  → render adapter returned None; image metrics skipped.")

    # 5) Save
    torch.save(
        {"texts": args.texts, "sequences": sequences,
         "utilisation": util,
         "image_metrics": image_metrics,
         "trajectory_lengths": [exec_out["final_state"].mu.shape[1]] * len(exec_out["trajectory"])},
        out_dir / "results.pt",
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "texts": args.texts,
            "n_sequences": int(sequences.shape[0]),
            "seq_len": int(sequences.shape[1]),
            "utilisation": util,
            "image_metrics": image_metrics,
        }, f, indent=2)
    print(f"  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
