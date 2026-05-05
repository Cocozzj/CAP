"""Diversity metrics for action generation (PDF lines 608, 1090).

Measures how diverse the generated atomic-token sequences are when
multiple samples are drawn from the same task token (Mode A with
``num_samples > 1``).

Metrics:
  - Levenshtein distance        (token-level edit distance, mean over pairs)
  - n-gram coverage             (unique n-grams / total n-grams)
  - KL divergence between trajectory codebook-usage distributions
  - Terminal-state variance     (||SceneState_final − mean||²)

Usage::
    python -m eval.diversity \\
        --ckpt runs/main_exp/ckpt/main_exp_final.pt \\
        --texts "open the drawer" \\
        --num-samples 16
"""

from __future__ import annotations

import argparse
import json
from typing import List

import torch

from model import build_scene_state
from dataloader import ToyDataset, collate_batch

from .utils import add_common_eval_args, load_model_for_eval, get_output_dir


# ──────────────────────────────────────────────────────────────────────
# Token-sequence metrics
# ──────────────────────────────────────────────────────────────────────

def levenshtein(a: List[int], b: List[int]) -> int:
    """Standard edit distance between two integer sequences (DP, O(|a|·|b|))."""
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j - 1], dp[j])
            prev = tmp
    return dp[m]


def mean_pairwise_levenshtein(seqs: List[List[int]]) -> float:
    """Mean Levenshtein distance over all unordered pairs."""
    if len(seqs) < 2:
        return 0.0
    total, count = 0, 0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            total += levenshtein(seqs[i], seqs[j])
            count += 1
    return total / count


def ngram_coverage(seqs: List[List[int]], n: int = 3) -> float:
    """Unique n-grams / total n-grams across all generated sequences.

    Higher = more diverse (1.0 means every n-gram is unique; 0.0 means full
    repetition).
    """
    grams = []
    for s in seqs:
        for i in range(max(len(s) - n + 1, 0)):
            grams.append(tuple(s[i:i + n]))
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def codebook_usage_kl(seqs: List[List[int]], num_codes: int) -> float:
    """KL(p_observed || p_uniform).  Lower = more uniform = higher diversity."""
    if not seqs:
        return 0.0
    flat = torch.tensor([t for s in seqs for t in s], dtype=torch.long)
    counts = torch.bincount(flat.clamp(0, num_codes - 1), minlength=num_codes).float()
    p = counts / counts.sum().clamp(min=1)
    q = torch.ones_like(p) / num_codes
    return float((p * (p.clamp(min=1e-12).log() - q.log())).sum())


# ──────────────────────────────────────────────────────────────────────
# Terminal-state variance (uses SceneState.mu)
# ──────────────────────────────────────────────────────────────────────

def terminal_state_variance(final_states: List) -> float:
    """Variance of final-state Gaussian centres across samples."""
    if len(final_states) < 2:
        return 0.0
    mus = torch.stack([s.mu for s in final_states], dim=0)        # [N, B, K, P, 3]
    # Variance per Gaussian, then mean over Gaussians
    return float(mus.var(dim=0, unbiased=False).mean())


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    add_common_eval_args(parser)
    parser.add_argument("--texts", nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=16,
                        help="How many samples to draw per text prompt")
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--enable-physics", action="store_true")
    args = parser.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "diversity")
    K_codes = model.encoder.action_enc.vq.num_codes

    print(f"\n=== Diversity metrics: {len(args.texts)} prompts × "
          f"{args.num_samples} samples ===")

    # Sample N action sequences per prompt
    plan_out = model.plan_from_text(texts=args.texts, num_samples=args.num_samples)
    sequences = plan_out["sequences"]    # [B*N, L_out]
    B = len(args.texts)
    N = args.num_samples
    sequences = sequences.view(B, N, -1)          # [B, N, L_out]

    per_prompt = {}

    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=B, sh_dim=sh_dim)
    batch = collate_batch([ds[i] for i in range(B)])
    gs_params = [g.to(device) for g in batch["gs_params"]]
    enc_out = model.encode(batch["frames"].to(device),
                                gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
    )

    for b, txt in enumerate(args.texts):
        seqs = [sequences[b, i].tolist() for i in range(N)]

        lev    = mean_pairwise_levenshtein(seqs)
        ngc    = ngram_coverage(seqs, n=args.ngram_n)
        kl     = codebook_usage_kl(seqs, K_codes)

        # Execute each sample to get final states
        finals = []
        with torch.no_grad():
            for i in range(N):
                K = scene.K
                plan_tok = model.unflatten_plan(sequences[b, i:i + 1], K=K)
                ppseq = model.tokens_to_physical_params(plan_tok)
                # Execute on a single-sample slice of the scene
                from model.utils import SceneState as _SS
                from model.utils import CanonicalFrame as _CF
                single = _SS(
                    mu=scene.mu[b:b + 1], cov=scene.cov[b:b + 1],
                    sh=scene.sh[b:b + 1], opacity=scene.opacity[b:b + 1],
                    scale=scene.scale[b:b + 1],
                    phi=_CF(R_w2c=scene.phi.R_w2c[b:b + 1],
                            t_w2c=scene.phi.t_w2c[b:b + 1]),
                    mask=scene.mask[b:b + 1],
                    R_obj_world=scene.R_obj_world[b:b + 1],
                )
                exec_out = model.execute_sequence(
                    scene=single, physical_params_seq=ppseq,
                    enable_physics=args.enable_physics,
                )
                finals.append(exec_out["final_state"])
        var = terminal_state_variance(finals)

        per_prompt[txt] = {
            "mean_pairwise_levenshtein": lev,
            f"ngram_{args.ngram_n}_coverage": ngc,
            "codebook_usage_KL_to_uniform": kl,
            "terminal_state_variance":    var,
        }
        print(f"  {txt!r}")
        print(f"    levenshtein={lev:.2f}  ngram-cov={ngc:.3f}  "
              f"KL={kl:.3f}  term-var={var:.4e}")

    summary = {
        "per_prompt":  per_prompt,
        "num_samples": N,
        "ngram_n":     args.ngram_n,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    torch.save(summary, out_dir / "results.pt")
    print(f"\n  ✔ saved to {out_dir}")


if __name__ == "__main__":
    main()
