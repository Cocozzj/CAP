"""MotionGPT inference: text → tokens → 4DGS.

SKELETON — wire to MotionGPT's actual generation API after cloning their repo.

Usage:

    python -m eval.baseline.motiongpt.infer \\
        --motiongpt-ckpt $MOTIONGPT_REPO/checkpoints/finetuned.ckpt \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --output-root runs/baselines
"""
from __future__ import annotations

import argparse
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
from ..flat_vqvae.data import delta_to_pose
from ..kinematics import apply_pose_trajectory_to_gs, quat_log_scale_to_full_cov


def _load_motiongpt_model(ckpt_path: Path, device):
    """Load fine-tuned MotionGPT (T5 + motion VQ).

    TODO: implement after cloning MotionGPT.  Real loading involves:
        from mgpt.models.mgpt import MotionGPT
        model = MotionGPT.load_from_checkpoint(ckpt_path)
        model.to(device).eval()
        return model
    """
    raise NotImplementedError(
        "MotionGPT loading is a TODO — see eval/baseline/motiongpt/README.md "
        "for setup instructions."
    )


def _generate_tokens(model, text: str, num_samples: int = 1) -> List[List[int]]:
    """Generate motion token sequences from text.

    TODO: implement using MotionGPT's text-to-motion API.
    """
    return []


def _tokens_to_pose_deltas(tokens: List[int]) -> Optional[np.ndarray]:
    """Decode token sequence → per-frame pose deltas [T-1, 7].

    TODO: implement using MotionGPT's motion VQ-VAE decoder.
    Or: re-use ours flat_vqvae VQ-VAE if vocab is shared.
    """
    return None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--motiongpt-ckpt", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["test_iid"])
    p.add_argument("--output-root",  default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        model = _load_motiongpt_model(Path(args.motiongpt_ckpt), device)
    except NotImplementedError as e:
        print(f"⚠ {e}", file=sys.stderr)
        print("  Marking all trajectories as 'motiongpt_not_implemented'.",
              file=sys.stderr)
        model = None

    n_total = n_ok = n_skip = 0
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

            text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
            out_dir = baseline_output_dir(
                args.output_root, "motiongpt",
                args.dataset_name, split, traj_id,
            )

            if model is None:
                TrajMetrics(notes="motiongpt_not_implemented").save(out_dir / "metrics.json")
                n_skip += 1
                continue

            try:
                samples = _generate_tokens(model, text, num_samples=1)
                if not samples:
                    raise RuntimeError("empty token sequence")
                deltas = _tokens_to_pose_deltas(samples[0])
                if deltas is None:
                    raise RuntimeError("delta decode failed")

                # Initial pose: from trajectory.npz if available, else identity
                pose0 = np.zeros(7, dtype=np.float32); pose0[6] = 1.0
                npz = traj_dir / "trajectory.npz"
                if npz.exists():
                    z = np.load(npz, allow_pickle=False)
                    if "object_pose_world" in z.files:
                        pose0 = z["object_pose_world"][0].astype(np.float32)
                poses = delta_to_pose(deltas, pose0)

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
                seq.save(out_dir / "pred_4dgs.npz")
                TrajMetrics(notes="pending_eval").save(out_dir / "metrics.json")
                n_ok += 1
                if n_ok <= 3 or n_ok % 50 == 0:
                    print(f"  ✓ {traj_id}")
            except Exception as e:
                print(f"  ✗ {traj_id}  {type(e).__name__}: {e}")
                n_skip += 1

    print(f"\n=== MotionGPT inference: ok={n_ok}, skip={n_skip}, "
          f"elapsed={time.time()-t0:.1f}s ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
