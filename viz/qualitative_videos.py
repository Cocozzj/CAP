"""
qualitative_videos.py — V.1 (CAP/Experiment.md §5 Fig 1).

Produce a 5 tasks × M methods grid of trajectory snapshots.  Each cell
is a horizontal strip of T evenly-spaced timesteps from one rollout.

Inputs
------
  --tasks          5 text instructions (e.g., "open the drawer", ...)
  --methods        named (label, ckpt) pairs:
                       "Ours:runs/main_exp/seed_0/ckpt/main_exp_final.pt"
                       "NoPlanner:runs/abl_no_planner/.../final.pt"
                       ...
  --timesteps      number of frames sampled from each trajectory (default 8)
  --output-dir     where to write grid.png + per-cell .npy
  --renderer       "auto" | "gsplat" | "scatter"  (auto: gsplat if installed,
                                                         else scatter fallback)

Output
------
  - ``grid.png``         — final figure (matplotlib)
  - ``cells/<task>_<method>.png``   — per-cell strips for paper supplementary
  - ``trajectories.pt``  — torch dump of mu trajectories per (task, method)

Renderer fallback
-----------------
If no 3DGS renderer is installed (or render_hook still TODO), we draw
each timestep as a scatter of Gaussian centres coloured by slot index.
This is enough to visually compare *what moved when* across methods —
swap in true gsplat rendering once render_hook is wired up.

Usage::

    python -m viz.qualitative_videos \\
        --tasks "open the drawer" "rotate the knob" "lift the cup" \\
                "press the lever" "pour the bottle" \\
        --methods Ours:runs/main_exp/seed_0/ckpt/main_exp_final.pt \\
                  NoPlanner:runs/abl_no_planner/seed_0/ckpt/main_exp_final.pt \\
                  NoPhys:runs/abl_no_physics/seed_0/ckpt/main_exp_final.pt \\
                  NoTask:runs/abl_no_task/seed_0/ckpt/main_exp_final.pt \\
                  PBDOnly:runs/abl_pbd_only/seed_0/ckpt/main_exp_final.pt \\
                  Baseline:runs/baseline/seed_0/ckpt/main_exp_final.pt \\
        --timesteps 8 \\
        --output-dir runs/figures/qualitative
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")                                  # no display required
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import build_scene_state
from dataloader import ToyDataset, collate_batch

from eval.utils import load_model_for_eval
from eval.render_hook import available_backend


# ──────────────────────────────────────────────────────────────────────
# Trajectory generation
# ──────────────────────────────────────────────────────────────────────

def _rollout(
    model, *,
    text: str,
    scene_batch,
    device,
    enable_physics: bool = True,
):
    """Run one text-conditioned rollout, return list of SceneState along trajectory."""
    gs_params = [g.to(device) for g in scene_batch["gs_params"]]
    enc_out = model.encode(scene_batch["frames"].to(device),
                           gs_params=gs_params, tau=1.0)
    scene = build_scene_state(
        gs_params=gs_params, phi=enc_out["phi"], assignment=enc_out["assignment"],
    )

    plan_out    = model.plan_from_text([text], num_samples=1)
    plan_tokens = model.unflatten_plan(plan_out["sequences"], K=scene.K)
    ppseq       = model.tokens_to_physical_params(plan_tokens)

    exec_out = model.execute_sequence(
        scene=scene, physical_params_seq=ppseq, enable_physics=enable_physics,
    )
    # Trajectory is list of SceneState per timestep; prepend initial.
    return [scene] + list(exec_out["trajectory"])


def _sample_timesteps(traj_len: int, T: int) -> List[int]:
    """Pick T indices evenly spaced (including first and last)."""
    if traj_len <= 1:
        return [0] * T
    if T == 1:
        return [traj_len - 1]
    return [round(i * (traj_len - 1) / (T - 1)) for i in range(T)]


# ──────────────────────────────────────────────────────────────────────
# Cell rendering
# ──────────────────────────────────────────────────────────────────────

def _scatter_cell(scenes, ax_strip, task_label: str, method_label: str,
                  show_x_label: bool):
    """Render T scenes as a horizontal strip of 3D scatter snapshots."""
    T = len(scenes)
    # Find a global axis range so all snapshots share scale.
    # IMPORTANT: only count REAL Gaussians — padded slots have arbitrary mu
    # and would distort the bounding box.
    real_xyz = []
    for s in scenes:
        mu_b   = s.mu[0].reshape(-1, 3).detach().cpu()                # [K*N, 3]
        mask_b = (s.mask[0].reshape(-1).detach().cpu().bool()
                  if s.mask is not None else
                  torch.ones(mu_b.shape[0], dtype=torch.bool))
        if mask_b.any():
            real_xyz.append(mu_b[mask_b])
    if real_xyz:
        all_xyz = torch.cat(real_xyz, dim=0)
    else:
        all_xyz = torch.zeros(1, 3)
    pad = 0.05
    xmin, ymin, zmin = (all_xyz.min(dim=0).values - pad).tolist()
    xmax, ymax, zmax = (all_xyz.max(dim=0).values + pad).tolist()

    K = scenes[0].mu.shape[1]
    cmap = matplotlib.colormaps.get_cmap("tab10")
    colors = [cmap(k % 10) for k in range(K)]

    for t, ax in zip(range(T), ax_strip):
        ax.cla()
        s = scenes[t]
        for k in range(K):
            mu = s.mu[0, k].detach().cpu().numpy()
            mask = (s.mask[0, k].detach().cpu().numpy()
                    if s.mask is not None else np.ones(mu.shape[0], dtype=bool))
            mu = mu[mask]
            if mu.shape[0] == 0:
                continue
            # Sub-sample to keep figure light.
            if mu.shape[0] > 200:
                idx = np.random.RandomState(0).choice(mu.shape[0], 200, replace=False)
                mu = mu[idx]
            ax.scatter(mu[:, 0], mu[:, 1], mu[:, 2],
                       s=2.0, c=[colors[k]], alpha=0.6, edgecolors="none")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1, 1, 1))
        if t == 0:
            ax.text2D(0.0, 1.05, method_label, transform=ax.transAxes,
                      fontsize=8, ha="left", va="bottom")
        if show_x_label:
            ax.set_xlabel(f"t={t}", fontsize=6, labelpad=-12)


# ──────────────────────────────────────────────────────────────────────
# Method spec parsing
# ──────────────────────────────────────────────────────────────────────

def _parse_methods(spec_list: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for spec in spec_list:
        if ":" not in spec:
            raise SystemExit(f"--methods entries must be 'Label:path/to/ckpt.pt' (got {spec!r})")
        label, ckpt = spec.split(":", 1)
        out.append((label.strip(), Path(ckpt.strip())))
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",       nargs="+", required=True,
                   help="List of text instructions (rows of grid)")
    p.add_argument("--methods",     nargs="+", required=True,
                   help="Each item 'Label:/path/to/ckpt.pt' (columns of grid)")
    p.add_argument("--timesteps",   type=int, default=8,
                   help="How many timestep frames per cell")
    p.add_argument("--output-dir",  type=str, default="runs/figures/qualitative")
    p.add_argument("--device",      type=str, default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--enable-physics", action="store_true", default=True)
    p.add_argument("--no-physics",  dest="enable_physics", action="store_false")
    p.add_argument("--renderer",    type=str, default="auto",
                   choices=["auto", "gsplat", "scatter"])
    p.add_argument("--scene-index", type=int, default=0,
                   help="Which ToyDataset index to use as the initial scene")
    args = p.parse_args()

    out = Path(args.output_dir); (out / "cells").mkdir(parents=True, exist_ok=True)

    methods = _parse_methods(args.methods)
    tasks   = list(args.tasks)
    T       = int(args.timesteps)
    n_rows  = len(tasks)
    n_cols  = len(methods)

    # Renderer choice
    if args.renderer == "auto":
        backend = available_backend()
        use_scatter = (backend is None)
    else:
        use_scatter = (args.renderer == "scatter")

    if use_scatter:
        print("Renderer: SCATTER fallback (no 3DGS rasterizer wired up).")
    else:
        print(f"Renderer: 3DGS via {available_backend()}")

    # ── Build grid figure: rows × (1 label col + n_cols × T cell columns) ──
    cell_w, cell_h = 1.0, 1.0           # per-timestep panel size in inches
    label_col_w = 1.2                   # reserved width for row labels
    fig_w = label_col_w + n_cols * (T * cell_w + 0.2)
    fig_h = n_rows * (cell_h + 0.4)
    fig = plt.figure(figsize=(fig_w, fig_h))
    # Total cols = 1 (label) + n_cols * T (snapshots).  width_ratios scales
    # the label column to ~label_col_w inches relative to per-cell width.
    n_cell_cols = n_cols * T
    width_ratios = [label_col_w / cell_w] + [1.0] * n_cell_cols
    gs = fig.add_gridspec(n_rows, n_cell_cols + 1,
                          width_ratios=width_ratios, wspace=0.05, hspace=0.4)

    traj_dump: Dict[str, Dict[str, list]] = {}

    for r, task in enumerate(tasks):
        traj_dump[task] = {}
        for c, (label, ckpt_path) in enumerate(methods):
            print(f"\n[{r * n_cols + c + 1}/{n_rows * n_cols}] task={task!r}  method={label}")
            # ── Load model lazily per cell ──
            ns = SimpleNamespace(
                ckpt=str(ckpt_path), config=None,
                device=args.device, output_dir=None,
            )
            model, cfg, dev = load_model_for_eval(ns)

            sh_dim = cfg["gs_param"]["gs_dimension"] - 11
            ds = ToyDataset(n_samples=args.scene_index + 1, sh_dim=sh_dim)
            scene_batch = collate_batch([ds[args.scene_index]])

            with torch.no_grad():
                traj = _rollout(model, text=task, scene_batch=scene_batch,
                                device=dev, enable_physics=args.enable_physics)
            idx_t = _sample_timesteps(len(traj), T)
            sampled = [traj[i] for i in idx_t]

            traj_dump[task][label] = [s.mu[0].detach().cpu() for s in sampled]

            # ── Render strip into the grid ──
            # gs columns: 0 = label, 1..n_cell_cols = snapshot cells.
            ax_strip = []
            for j in range(T):
                ax = fig.add_subplot(gs[r, 1 + c * T + j], projection="3d")
                ax_strip.append(ax)
            _scatter_cell(sampled, ax_strip,
                          task_label=task,
                          method_label=label if r == 0 else "",
                          show_x_label=(r == n_rows - 1))

            # Per-cell standalone PNG (paper supplementary)
            cell_fig = plt.figure(figsize=(T * cell_w, cell_h))
            cell_axes = [cell_fig.add_subplot(1, T, j + 1, projection="3d")
                         for j in range(T)]
            _scatter_cell(sampled, cell_axes,
                          task_label=task, method_label=label,
                          show_x_label=True)
            safe_task   = task.replace(" ", "_").replace("/", "_")
            safe_method = label.replace(" ", "_").replace("/", "_")
            cell_path   = out / "cells" / f"{safe_task}__{safe_method}.png"
            cell_fig.savefig(cell_path, dpi=200, bbox_inches="tight")
            plt.close(cell_fig)
            print(f"   ✔ cell → {cell_path.name}")

            # Free model before next iteration
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Row label in the dedicated label column (gs[:, 0])
        row_ax = fig.add_subplot(gs[r, 0])
        row_ax.axis("off")
        row_ax.text(0.95, 0.5, task, transform=row_ax.transAxes,
                    fontsize=9, va="center", ha="right", rotation=0,
                    wrap=True)

    grid_path = out / "grid.png"
    fig.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    torch.save(traj_dump, out / "trajectories.pt")
    print(f"\n✓ Wrote {grid_path}  and  {out / 'trajectories.pt'}")


if __name__ == "__main__":
    main()
