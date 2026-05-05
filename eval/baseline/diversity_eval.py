"""Action Diversity (PDF metric #9): for each text instruction, sample N
sequences, compute average pairwise normalized edit distance.

Formula (PDF section 5.1):

    D = (1 / (N(N-1))) * Σ_{i<j} edit_dist(seq_i, seq_j) / max(|seq_i|, |seq_j|)
    D ∈ [0, 1]   higher = more diverse

For each baseline that supports stochastic sampling (Flat VQ-VAE, MAGVIT-v2,
Ours), we run N inferences per text input and aggregate.

Deterministic baselines (TAMP-rule, PhysGaussian, 4D-GS) are skipped — they
produce identical outputs every time, so D = 0 trivially.

Output: writes ``diversity.json`` per (baseline, dataset, split) with
mean / std of D across trajectories, plus per-trajectory raw values.

Usage:

    # First make sure the baseline supports multi-sample inference.
    # This script INVOKES the baseline's infer module N times per text.
    python -m eval.baseline.diversity_eval \\
        --baselines flat_vqvae ours \\
        --manifest dataset/dataset_a/manifest.json \\
        --data-dir dataset/dataset_a/data \\
        --splits test_iid \\
        --N 10 \\
        --output-root runs/baselines
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence

import numpy as np


# ══════════════════════════════════════════════════════════════════════
# Edit distance (Levenshtein)
# ══════════════════════════════════════════════════════════════════════

def edit_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """Compute Levenshtein edit distance between two integer sequences.

    Standard O(|a|·|b|) DP.  Token sequences are usually short (T-1 ≈ 30),
    so this is plenty fast.
    """
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    cur  = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,                       # insertion
                prev[j] + 1,                          # deletion
                prev[j - 1] + cost,                   # substitution
            )
        prev, cur = cur, prev
    return prev[len(b)]


def pairwise_diversity(samples: List[Sequence[int]]) -> float:
    """PDF formula: D = (1 / N(N-1)) * Σ_{i<j} edit_dist / max(|i|, |j|).

    Note: the formula has 1/(N(N-1)) which counts ordered pairs.
    With i<j we have N(N-1)/2 unordered pairs, multiplied by 2 because
    we sum both (i,j) and (j,i) — equivalent to using 1/(N(N-1)).
    Implemented as 2 * Σ_{i<j} / (N*(N-1))  =  Σ_{i<j} / C(N,2).
    """
    N = len(samples)
    if N < 2:
        return 0.0
    total = 0.0
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            ed = edit_distance(samples[i], samples[j])
            denom = max(len(samples[i]), len(samples[j]), 1)
            total += ed / denom
            n_pairs += 1
    return float(total / max(n_pairs, 1))


# ══════════════════════════════════════════════════════════════════════
# Per-baseline multi-sample runners
# ══════════════════════════════════════════════════════════════════════
# Each baseline that supports stochastic sampling has its own runner; the
# runner returns N token sequences for a given text.  The runner re-uses
# already-trained checkpoints — this script does NOT train.

def _samples_flat_vqvae(text, args, ckpts, device, N: int) -> List[List[int]]:
    """Sample N times from Flat VQ-VAE for a single text."""
    import torch
    from sentence_transformers import SentenceTransformer
    from eval.baseline.flat_vqvae.gpt   import FlatGPT
    from eval.baseline.flat_vqvae.vqvae import FlatVQVAE

    # Lazy load (memoized)
    if "flat_vqvae_models" not in ckpts:
        vq_state = torch.load(args.flat_vqvae_ckpt, map_location=str(device))
        vq_args = vq_state.get("args", {})
        vqvae = FlatVQVAE(
            in_dim=7,
            hidden=vq_args.get("hidden", 128),
            code_dim=vq_args.get("code_dim", 32),
            K=vq_args.get("K", 64),
        ).to(device)
        vqvae.load_state_dict(vq_state["model"]); vqvae.eval()

        gpt_state = torch.load(args.flat_vqvae_gpt_ckpt, map_location=str(device))
        gpt_args = gpt_state.get("args", {})
        gpt = FlatGPT(
            vocab_size=gpt_args.get("K", 64),
            d_model=gpt_args.get("gpt_d_model", 128),
            n_layers=gpt_args.get("gpt_n_layers", 3),
            n_heads=gpt_args.get("gpt_n_heads", 4),
            d_ff=gpt_args.get("gpt_d_ff", 512),
            d_text=768, max_seq_len=args.T - 1,
        ).to(device)
        gpt.load_state_dict(gpt_state["model"]); gpt.eval()

        text_enc = SentenceTransformer("all-mpnet-base-v2").to(device).eval()
        ckpts["flat_vqvae_models"] = (vqvae, gpt, text_enc)

    vqvae, gpt, text_enc = ckpts["flat_vqvae_models"]
    out: List[List[int]] = []
    with torch.no_grad():
        emb = text_enc.encode([text], convert_to_tensor=True, device=str(device)).float()
        for _ in range(N):
            ids = gpt.generate(emb, max_tokens=args.T - 1,
                                temperature=args.temperature, top_k=args.top_k)
            out.append(ids[0].cpu().tolist())
    return out


def _samples_ours(text, args, ckpts, device, N: int) -> List[List[int]]:
    """Sample N times from Ours for a single text.

    Uses ``eval.baseline.ours.runner.sample_action_tokens`` which calls
    ``model.plan_from_text(num_samples=N)`` and returns N token id sequences.
    """
    if "ours_model" not in ckpts:
        try:
            from eval.baseline.ours.runner import load_model
            ckpts["ours_model"] = load_model(args.ours_ckpt, args.ours_config, device)
        except Exception as e:
            print(f"[ours] failed to load model: {e}", file=sys.stderr)
            return []
    model = ckpts["ours_model"]
    try:
        from eval.baseline.ours.runner import sample_action_tokens
        return sample_action_tokens(model, text, num_samples=N)
    except Exception as e:
        print(f"[ours] sampling failed: {e}", file=sys.stderr)
        return []


def _samples_magvit(text, args, ckpts, device, N: int) -> List[List[int]]:
    """Sample N times from MAGVIT-v2 transformer for a single text.

    NOTE: MAGVIT-v2 produces video tokens (a longer sequence than action
    tokens).  Edit distance over MAGVIT tokens is meaningful but not directly
    comparable in magnitude to flat-VQVAE / Ours action tokens.  We report
    them separately or normalize.
    """
    print("[magvit_v2] multi-sample generation requires MaskGit inference loop "
           "with different seeds; not yet implemented in this script. "
           "Adjust eval/baseline/magvit_v2/infer.py to expose num_samples.",
           file=sys.stderr)
    return []


_SAMPLE_FN = {
    "ours":        _samples_ours,
    "magvit_v2":   _samples_magvit,
    "motiongpt":   _samples_magvit,         # TODO: replace with MotionGPT-specific sampler
    "flat_vqvae":  _samples_flat_vqvae,     # legacy / ablation use
    # PhysDreamer is stochastic via diffusion sampling but very expensive
    # — implement a sampler if you want diversity numbers for it.
}

_DETERMINISTIC = {"tamp_pddl", "physgaussian", "tamp_rule", "_4dgs"}


# ══════════════════════════════════════════════════════════════════════
# Main aggregation
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+",
                   default=["ours", "magvit_v2", "motiongpt"],
                   help="only baselines that support stochastic sampling are evaluated; "
                        "deterministic ones (TAMP, PhysGaussian) get D=0 by definition.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--splits", nargs="+", default=["test_iid"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--N", type=int, default=10,
                   help="samples per text (PDF default: 10)")
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k",       type=int, default=50)
    p.add_argument("--limit", type=int, default=None,
                   help="for debugging, only do N trajectories per split")

    # Per-baseline ckpt args — pass the ckpts the runners need
    p.add_argument("--flat-vqvae-ckpt",     default=None)
    p.add_argument("--flat-vqvae-gpt-ckpt", default=None)
    p.add_argument("--ours-ckpt",           default=None)
    p.add_argument("--ours-config",         default="configs/config.yaml")

    args = p.parse_args(argv)

    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from .common import baseline_output_dir, iter_split_entries
    from dataload.text import task_to_text

    ckpts: Dict = {}
    n_total = 0
    t0 = time.time()

    for baseline in args.baselines:
        is_det = baseline in _DETERMINISTIC
        if is_det:
            print(f"\n=== {baseline}: deterministic, D=0 by construction ===")
            # Write a stub diversity.json so the aggregator picks up D=0
            for split in args.splits:
                out_dir = Path(args.output_root) / baseline / args.dataset_name / split
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "diversity.json", "w") as f:
                    json.dump({"D_mean": 0.0, "D_std": 0.0, "n_trajs": 0,
                                "note": "deterministic baseline"}, f, indent=2)
            continue

        sample_fn = _SAMPLE_FN.get(baseline)
        if sample_fn is None:
            print(f"\n=== {baseline}: no sampler implemented, skipping ===")
            continue

        for split in args.splits:
            print(f"\n=== {baseline}  {args.dataset_name}/{split}  N={args.N} ===")
            n_split = 0
            per_traj_D: List[float] = []
            for traj_id, traj_dir, entry in iter_split_entries(
                args.manifest, args.data_dir, split,
            ):
                if args.limit is not None and n_split >= args.limit:
                    break
                n_split += 1
                n_total += 1
                text = task_to_text(entry["task_name"], entry.get("obj_category", ""))
                samples = sample_fn(text, args, ckpts, device, args.N)
                if len(samples) < 2:
                    continue
                D = pairwise_diversity(samples)
                per_traj_D.append(D)
                if n_split <= 3 or n_split % 50 == 0:
                    print(f"  {traj_id}  D={D:.3f}  ({len(samples)} samples)")

            # Aggregate D across trajectories
            if per_traj_D:
                D_mean = float(mean(per_traj_D))
                D_std  = float(stdev(per_traj_D)) if len(per_traj_D) > 1 else 0.0
            else:
                D_mean = D_std = float("nan")

            out_dir = Path(args.output_root) / baseline / args.dataset_name / split
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "diversity.json", "w") as f:
                json.dump({
                    "D_mean":  D_mean,
                    "D_std":   D_std,
                    "n_trajs": len(per_traj_D),
                    "N_per_traj": args.N,
                    "per_traj_D": per_traj_D,
                }, f, indent=2)
            print(f"  → split={split}  D = {D_mean:.4f} ± {D_std:.4f}  "
                  f"(over {len(per_traj_D)} trajs)")

    print(f"\nDiversity eval done in {time.time()-t0:.1f}s, {n_total} traj-samples total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
