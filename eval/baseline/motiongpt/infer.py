"""MotionGPT inference: text → T5 generate → motion ids → 4DGS.

Pipeline per trajectory:
  1. text → T5 → output string with <m_X> tokens
  2. Parse motion ids from output
  3. VQ-VAE.decode → pose deltas
  4. Integrate deltas → object_pose_world trajectory
  5. Apply trajectory to init_gs → 4DGS sequence
  6. Save pred_4dgs.npz

Usage:

    python -m eval.baseline.motiongpt.infer \\
        --motiongpt-dir runs/baselines/motiongpt/dataset_a/t5 \\
        --vqvae-ckpt    runs/baselines/flat_vqvae/dataset_a/vqvae/ckpt_final.pt \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits   test_iid \\
        --output-root runs/baselines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from dataload.common import load_init_gs_ply
from dataload.text import task_to_text

from ..common import (
    GS4DSequence,
    TrajMetrics,
    baseline_output_dir,
    iter_split_entries,
)
from ..kinematics import apply_pose_trajectory_to_gs, quat_log_scale_to_full_cov
from .data  import MGSpecialTokens, format_input_text, parse_motion_ids_from_text, delta_to_pose
from .vqvae import FlatVQVAE


# ──────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────

def load_motiongpt(motiongpt_dir: Path, device):
    """Load fine-tuned T5 + tokenizer + paired VQ-VAE config."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        raise SystemExit("transformers not installed")

    tokenizer = T5Tokenizer.from_pretrained(str(motiongpt_dir))
    model     = T5ForConditionalGeneration.from_pretrained(str(motiongpt_dir)).to(device)
    model.eval()

    # Load extra config (VQ-VAE ckpt path + K)
    extra_path = motiongpt_dir / "config_extra.json"
    if not extra_path.exists():
        raise FileNotFoundError(f"missing {extra_path} — did training save it?")
    with open(extra_path) as f:
        extra = json.load(f)
    return model, tokenizer, extra


def load_vqvae_from_extra(extra: dict, device) -> FlatVQVAE:
    vq_state = torch.load(extra["vqvae_ckpt"], map_location=str(device))
    vq_args  = vq_state.get("args", {})
    vqvae = FlatVQVAE(
        in_dim=7,
        hidden=vq_args.get("hidden", 128),
        code_dim=vq_args.get("code_dim", 32),
        K=vq_args.get("K", 64),
    ).to(device)
    vqvae.load_state_dict(vq_state["model"])
    vqvae.eval()
    return vqvae


# ──────────────────────────────────────────────────────────────────────
# Generation helper
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_motion_ids(
    model, tokenizer, text: str,
    max_length: int = 64,
    num_beams:  int = 4,
    num_samples: int = 1,
    do_sample:   bool = False,
    device=None,
) -> List[List[int]]:
    """Run T5 inference; return num_samples motion-id sequences (list of lists)."""
    src = format_input_text(text)
    enc = tokenizer([src], padding=True, truncation=True, return_tensors="pt").to(device)
    out = model.generate(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=num_samples,
        do_sample=do_sample,
        early_stopping=True,
    )
    decoded = [tokenizer.decode(s, skip_special_tokens=False) for s in out]
    return [parse_motion_ids_from_text(d) for d in decoded]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--motiongpt-dir", required=True,
                   help="dir from train.py (contains T5 weights + tokenizer + config_extra)")
    p.add_argument("--vqvae-ckpt", default=None,
                   help="override VQ-VAE ckpt path (default: from config_extra.json)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["test_iid"])
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"⏬ loading MotionGPT from {args.motiongpt_dir}")
    model, tokenizer, extra = load_motiongpt(Path(args.motiongpt_dir), device)
    if args.vqvae_ckpt:
        extra["vqvae_ckpt"] = args.vqvae_ckpt
    vqvae = load_vqvae_from_extra(extra, device)

    n_total = n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        n_split = 0
        for traj_id, traj_dir, entry in iter_split_entries(
            args.manifest, args.data_dir, split,
        ):
            if args.limit is not None and n_split >= args.limit:
                break
            n_split += 1
            n_total += 1

            out_dir = baseline_output_dir(
                args.output_root, "motiongpt",
                args.dataset_name, split, traj_id,
            )
            pred_path = out_dir / "pred_4dgs.npz"
            if args.skip_existing and pred_path.exists():
                n_skip += 1
                continue

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            try:
                # 1) Generate motion ids via T5
                samples = generate_motion_ids(
                    model, tokenizer, text,
                    max_length=args.T + 8, num_beams=args.num_beams,
                    num_samples=1, do_sample=False, device=device,
                )
                if not samples or not samples[0]:
                    raise RuntimeError("T5 generated empty motion sequence")
                ids = samples[0]
                # Truncate / pad to exactly T-1
                if len(ids) >= args.T - 1:
                    ids = ids[: args.T - 1]
                else:
                    ids = ids + [ids[-1]] * (args.T - 1 - len(ids))

                # 2) Decode motion ids → pose deltas via FlatVQVAE
                with torch.no_grad():
                    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
                    deltas = vqvae.decode_from_ids(ids_tensor)[0].cpu().numpy()    # [T-1, 7]

                # 3) Integrate deltas → absolute pose
                pose0 = np.zeros(7, dtype=np.float32); pose0[6] = 1.0
                npz = traj_dir / "trajectory.npz"
                if npz.exists():
                    z = np.load(npz, allow_pickle=False)
                    if "object_pose_world" in z.files:
                        pose0 = z["object_pose_world"][0].astype(np.float32)
                poses = delta_to_pose(pose0, deltas)

                # 4) Apply pose trajectory → 4DGS
                gs = load_init_gs_ply(traj_dir / "init_gs.ply",
                                       n_points=10000, seed=0, c_sh=48)
                mu0      = gs.mu.numpy().astype(np.float32)
                cov0     = quat_log_scale_to_full_cov(gs.cov.numpy(), gs.scale.numpy())
                sh0      = gs.sh.numpy().astype(np.float32)
                opacity0 = gs.opacity.numpy().astype(np.float32)
                scale0   = gs.scale.numpy().astype(np.float32)
                mu_t, cov_t, sh_t, opacity_t, scale_t = apply_pose_trajectory_to_gs(
                    mu0, cov0, sh0, opacity0, scale0, poses=poses,
                )
                seq = GS4DSequence(mu=mu_t, cov=cov_t, sh=sh_t,
                                   opacity=opacity_t, scale=scale_t)

                seq.save(pred_path)
                np.save(out_dir / "pred_tokens.npy", np.array(ids, dtype=np.int64))
                TrajMetrics(notes="pending_eval").save(out_dir / "metrics.json")
                n_ok += 1
                if n_ok <= 3 or n_ok % 50 == 0:
                    print(f"  ✓ {traj_id}  T={seq.T} N={seq.N} tokens={len(ids)}")
            except Exception as e:
                print(f"  ✗ {traj_id}  {type(e).__name__}: {e}")
                TrajMetrics(notes=f"motiongpt_failed: {type(e).__name__}").save(
                    out_dir / "metrics.json")
                n_fail += 1

    print(f"\n=== MotionGPT inference complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skip}")
    print(f"  failed:  {n_fail}")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
