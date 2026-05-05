"""
tsne.py — Fig 4 (CAP/Experiment.md §5).

Two-panel t-SNE of the action codebook:

  Panel A: one point per atomic code (K_action × token_dim), coloured by
           the *dominant task* that uses it most often across the eval set.
  Panel B: same points, coloured by the *dominant slot* (object) the code
           is dispatched to.  Together they show whether the codebook is
           organised by task semantics, by per-object physics, or both.

If the model has a non-trivial task codebook (use_task_token=True), we
also drop in a third panel: t-SNE of the task codebook (J × task_dim).

Usage::

    python -m viz.tsne \\
        --ckpt runs/main_exp/seed_0/ckpt/main_exp_final.pt \\
        --tasks "open the drawer" "rotate the knob" "lift the cup" \\
                "press the lever" "pour the bottle" "fold the cloth" \\
        --n-trials 16 \\
        --output-dir runs/figures/tsne
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.manifold import TSNE

from dataloader import ToyDataset, collate_batch

from eval.utils import add_common_eval_args, load_model_for_eval, get_output_dir


# ──────────────────────────────────────────────────────────────────────
# Dominant-label assignment
# ──────────────────────────────────────────────────────────────────────

def _dominant(label_grid: np.ndarray, n_codes: int, n_labels: int) -> np.ndarray:
    """Per-code dominant label.  ``label_grid[i, ℓ]`` = count of code i used
    by label ℓ.  Returns vector ``[n_codes]`` of dominant ℓ (or -1 if unused).
    """
    out = np.full(n_codes, -1, dtype=np.int64)
    for i in range(n_codes):
        if label_grid[i].sum() == 0:
            continue
        out[i] = int(label_grid[i].argmax())
    return out


# ──────────────────────────────────────────────────────────────────────
# Scatter helper
# ──────────────────────────────────────────────────────────────────────

def _scatter_panel(
    ax, embed: np.ndarray, labels: np.ndarray, label_names: List[str],
    title: str,
):
    """Plot 2D embedding, colour by integer label.  ``-1`` rendered grey."""
    cmap = matplotlib.colormaps.get_cmap("tab10")

    used = sorted(set(int(x) for x in labels if x >= 0))
    for ℓ in used:
        mask = (labels == ℓ)
        ax.scatter(embed[mask, 0], embed[mask, 1],
                   s=14, c=[cmap(ℓ % 10)], alpha=0.8,
                   edgecolors="none",
                   label=label_names[ℓ] if ℓ < len(label_names) else f"L{ℓ}")
    unused = (labels == -1)
    if unused.any():
        ax.scatter(embed[unused, 0], embed[unused, 1],
                   s=8, c="lightgrey", alpha=0.4, edgecolors="none",
                   label="(unused)")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6, loc="best", frameon=False, ncol=2)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    add_common_eval_args(p)
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Texts to drive the planner; codes used per text are tallied")
    p.add_argument("--n-trials", type=int, default=16,
                   help="Initial scenes per task (more = more reliable counts)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    model, cfg, device = load_model_for_eval(args)
    out_dir = get_output_dir(args, "tsne")

    # ── Codebooks ──
    atomic_W = model.atomic_codebook.detach().cpu().numpy()       # [K_action, token_dim]
    K_action = atomic_W.shape[0]
    print(f"\n=== t-SNE ===")
    print(f"  atomic codebook : {atomic_W.shape}")

    tcb = model.task_codebook                                     # call the property ONCE
    task_W = tcb.detach().cpu().numpy() if tcb is not None else None
    if task_W is not None:
        print(f"  task   codebook : {task_W.shape}")
    else:
        print(f"  task   codebook : (use_task_token=False — task panel skipped)")

    # ── Tally code usage by (task, slot) over a small dataset sweep ──
    sh_dim = cfg["gs_param"]["gs_dimension"] - 11
    ds = ToyDataset(n_samples=args.n_trials, sh_dim=sh_dim)

    # Need K (object slots) — pull from one trial.
    sample_batch = collate_batch([ds[0]])
    enc_out = model.encode(sample_batch["frames"].to(device),
                           gs_params=[g.to(device) for g in sample_batch["gs_params"]],
                           tau=1.0)
    K_slots = enc_out["seq_tokens"].shape[2]
    print(f"  K_slots         : {K_slots}")
    print(f"  tasks           : {args.tasks}")
    print(f"  n_trials        : {args.n_trials}")

    by_task = np.zeros((K_action, len(args.tasks)), dtype=np.int64)
    by_slot = np.zeros((K_action, K_slots),        dtype=np.int64)

    with torch.no_grad():
        for ti, txt in enumerate(args.tasks):
            for trial in range(args.n_trials):
                batch     = collate_batch([ds[trial]])
                gs_params = [g.to(device) for g in batch["gs_params"]]
                # We just need a scene to size K — the codes themselves come
                # from plan_from_text (which doesn't see the scene), but
                # unflatten_plan needs K to reshape.
                enc_o = model.encode(batch["frames"].to(device),
                                     gs_params=gs_params, tau=1.0)
                Kc = enc_o["seq_tokens"].shape[2]

                plan_out = model.plan_from_text([txt], num_samples=1)
                tokens   = model.unflatten_plan(plan_out["sequences"], K=Kc)  # [1, T, K]
                tokens_np = tokens[0].detach().cpu().numpy()                  # [T, K]

                T_eff = tokens_np.shape[0]
                for k in range(min(Kc, K_slots)):
                    for t in range(T_eff):
                        code = int(tokens_np[t, k])
                        if 0 <= code < K_action:
                            by_task[code, ti] += 1
                            by_slot[code, k]  += 1

    used_total = int((by_task.sum(axis=1) > 0).sum())
    print(f"  codes touched   : {used_total} / {K_action}  "
          f"({100.0 * used_total / K_action:.1f}%)")

    # ── Dominant label per code ──
    dom_task = _dominant(by_task, K_action, len(args.tasks))
    dom_slot = _dominant(by_slot, K_action, K_slots)

    # ── t-SNE on atomic codebook ──
    perplex = float(min(args.perplexity, max(5, K_action // 4)))
    tsne_atom = TSNE(
        n_components=2, perplexity=perplex,
        init="pca", metric="euclidean",
        random_state=args.seed,
    ).fit_transform(atomic_W)

    # ── Figure: 2 (or 3) panels ──
    n_panels = 2 + (1 if task_W is not None else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5.0))
    if n_panels == 1:
        axes = [axes]

    _scatter_panel(axes[0], tsne_atom, dom_task, args.tasks,
                   title="A. Atomic codes  (coloured by dominant task)")
    _scatter_panel(axes[1], tsne_atom, dom_slot,
                   [f"slot {k}" for k in range(K_slots)],
                   title="B. Atomic codes  (coloured by dominant slot)")

    if task_W is not None:
        J = task_W.shape[0]
        perplex_t = float(min(args.perplexity, max(2, J // 4)))
        if J >= 4:
            tsne_task = TSNE(
                n_components=2, perplexity=perplex_t,
                init="pca", random_state=args.seed,
            ).fit_transform(task_W)
        else:
            # Too few task codes for t-SNE — fall back to identity layout.
            tsne_task = task_W[:, :2] if task_W.shape[1] >= 2 \
                else np.column_stack([np.arange(J), np.zeros(J)])
        # Colour by index since we don't know task→J mapping deterministically.
        _scatter_panel(
            axes[2], tsne_task, np.arange(J),
            [f"task {j}" for j in range(J)],
            title=f"C. Task codes  (J={J})",
        )

    fig.tight_layout()
    fig_path = out_dir / "tsne.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Persist raw embeddings + counts for re-plotting
    np.savez(
        out_dir / "tsne_data.npz",
        atomic_W=atomic_W, tsne_atomic=tsne_atom,
        dom_task=dom_task, dom_slot=dom_slot,
        by_task=by_task,   by_slot=by_slot,
        task_W=task_W if task_W is not None else np.empty(0),
        tasks=np.array(args.tasks),
    )
    with open(out_dir / "summary.json", "w") as f:
        json.dump({
            "K_action":         int(K_action),
            "K_slots":          int(K_slots),
            "tasks":            args.tasks,
            "n_trials":         args.n_trials,
            "perplexity":       perplex,
            "codes_touched":    int(used_total),
            "task_codebook":    None if task_W is None else list(task_W.shape),
        }, f, indent=2)

    print(f"\n  ✔ t-SNE → {fig_path}")
    print(f"  ✔ data  → {out_dir / 'tsne_data.npz'}")


if __name__ == "__main__":
    main()
