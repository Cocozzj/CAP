"""
render_rollout_frames.py — Render method rollouts into PNG frames.

For each (task, method) pair:
  1. Load the method's checkpoint.
  2. Roll out the method on the given task (text instruction).
  3. Sample T timesteps evenly from the trajectory.
  4. Render each timestep with a fixed camera viewpoint into PNG.

Output structure (consumed by figure_qualitative_comparison.py)::

    <output_dir>/
        <task_id>/
            <method_name>/
                t000.png
                t002.png
                ...

Renderer
--------
Uses 3DGS rasterizer (gsplat / nerfacc) when available.  Falls back to
3D scatter projected to 2D so the layout still works without a full
rasterizer wired up.

Example
-------
    python -m eval.viz.render_rollout_frames \\
        --tasks open_drawer_unseen:"open the drawer" \\
                long_horizon_5step:"open then close then rotate" \\
        --methods Ours:runs/main_exp/seed_0/ckpt/main_exp_final.pt \\
                  PhysGaussian:runs/baseline_physgaussian/ckpt/final.pt \\
        --timesteps 0 2 4 6 8 10 \\
        --camera configs/cameras/default.yaml \\
        --output-dir runs/figures/qualitative/frames
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from model import build_scene_state
from dataload import collate_batch

from ..utils import add_data_args, build_eval_loader, load_model_for_eval
from ..render_hook import available_backend, render_scene


# ──────────────────────────────────────────────────────────────────────
# Trajectory rollout
# ──────────────────────────────────────────────────────────────────────

def rollout_trajectory(
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
        gs_params=gs_params,
        phi=enc_out["phi"],
        assignment=enc_out["assignment"],
    )

    plan_out = model.plan_from_text([text], num_samples=1)
    plan_tokens = model.unflatten_plan(plan_out["sequences"], K=scene.K)
    ppseq = model.tokens_to_physical_params(plan_tokens)

    exec_out = model.execute_sequence(
        scene=scene,
        physical_params_seq=ppseq,
        enable_physics=enable_physics,
    )
    return [scene] + list(exec_out["trajectory"])


def sample_timesteps(traj_len: int, indices: List[int]) -> List[int]:
    """Map requested step indices into valid trajectory positions."""
    if traj_len <= 1:
        return [0] * len(indices)
    return [min(t, traj_len - 1) for t in indices]


# ──────────────────────────────────────────────────────────────────────
# Camera + rendering
# ──────────────────────────────────────────────────────────────────────

def load_camera(camera_path: Path | None) -> dict:
    """Load camera intrinsics + extrinsics.  Returns a sensible default if missing."""
    if camera_path is not None and camera_path.exists():
        import yaml
        with open(camera_path) as f:
            return yaml.safe_load(f)
    # Default: front-ish viewpoint, 60deg FOV, 256x256
    return {
        "intrinsics": {
            "fx": 222.0, "fy": 222.0, "cx": 128.0, "cy": 128.0,
            "image_size": [256, 256],
        },
        "extrinsics": {
            # camera-to-world; looking at origin from (1.5, 1.0, 1.5)
            "translation": [1.5, 1.0, 1.5],
            "look_at":     [0.0, 0.0, 0.0],
            "up":          [0.0, 1.0, 0.0],
        },
    }


def render_scene_to_png(
    scene,
    camera: dict,
    output_path: Path,
    image_size: Tuple[int, int] = (256, 256),
) -> None:
    """Render one SceneState into a 2D PNG using the available backend.

    Falls back to 2D scatter (orthographic projection of the X-Y plane)
    if no 3DGS rasterizer is wired up.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    backend = available_backend()
    if backend is not None:
        # Use the real 3DGS rasterizer
        try:
            rendered = render_scene(scene, camera, image_size=image_size)
            if rendered is not None:
                # rendered: [V, 3, H, W] in [0, 1]
                img = (rendered[0].clamp(0, 1).cpu().numpy()
                       .transpose(1, 2, 0) * 255).astype(np.uint8)
                Image.fromarray(img).save(output_path)
                return
        except Exception as e:
            print(f"  (render fallback for {output_path.name}: {e})")

    # ── Scatter fallback ──
    H, W = image_size
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("white")

    mu = scene.mu[0].detach().cpu().reshape(-1, 3).numpy()       # [K*N, 3]
    if scene.mask is not None:
        mask = scene.mask[0].detach().cpu().reshape(-1).bool().numpy()
        mu = mu[mask]
    if mu.shape[0] > 0:
        # Subsample for figure clarity
        if mu.shape[0] > 800:
            idx = np.random.RandomState(0).choice(mu.shape[0], 800, replace=False)
            mu = mu[idx]
        # Simple orthographic: (x, y) in image plane
        ax.scatter(mu[:, 0], mu[:, 1], s=4, c="#3070C0", alpha=0.6,
                   edgecolors="none")
        # Set axis limits with padding
        pad = 0.1
        xmin, ymin = mu[:, :2].min(0) - pad
        xmax, ymax = mu[:, :2].max(0) + pad
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")

    fig.savefig(output_path, dpi=100, bbox_inches=None, pad_inches=0)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Spec parsing
# ──────────────────────────────────────────────────────────────────────

def parse_methods(spec_list: List[str]) -> List[Tuple[str, Path]]:
    out = []
    for spec in spec_list:
        if ":" not in spec:
            raise SystemExit(
                f"--methods entries must be 'Label:/path/to/ckpt.pt' (got {spec!r})"
            )
        label, ckpt = spec.split(":", 1)
        out.append((label.strip(), Path(ckpt.strip())))
    return out


def parse_tasks(spec_list: List[str]) -> List[Tuple[str, str]]:
    """Parse 'task_id:text instruction' specs."""
    out = []
    for s in spec_list:
        if ":" not in s:
            raise SystemExit(
                f"--tasks entries must be 'task_id:text instruction' (got {s!r})"
            )
        tid, instr = s.split(":", 1)
        out.append((tid.strip(), instr.strip()))
    return out


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Each item 'task_id:text instruction' "
                        "(e.g., 'open_drawer_unseen:open the drawer').")
    p.add_argument("--methods", nargs="+", required=True,
                   help="Each item 'Label:/path/to/ckpt.pt'.")
    p.add_argument("--timesteps", nargs="+", type=int,
                   default=[0, 2, 4, 6, 8, 10],
                   help="Timestep indices to render.")
    p.add_argument("--camera", type=Path, default=None,
                   help="YAML file with camera intrinsics + extrinsics.")
    p.add_argument("--image-size", type=int, nargs=2, default=[256, 256],
                   help="Output image size (H W).")
    p.add_argument("--output-dir", type=Path,
                   default=Path("runs/figures/qualitative/frames"))
    p.add_argument("--scene-index", type=int, default=0,
                   help="Which DatasetA index to use as the initial scene.")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"])
    p.add_argument("--enable-physics", action="store_true", default=True)
    p.add_argument("--no-physics", dest="enable_physics", action="store_false")
    add_data_args(p, default_split="val")
    args = p.parse_args()

    tasks = parse_tasks(args.tasks)
    methods = parse_methods(args.methods)
    timesteps = list(args.timesteps)
    image_size = tuple(args.image_size)
    camera = load_camera(args.camera)

    print(f"Renderer: {available_backend() or 'SCATTER fallback'}")
    print(f"Tasks:    {len(tasks)}; Methods: {len(methods)}; "
          f"Timesteps: {len(timesteps)}")

    for task_id, text in tasks:
        for label, ckpt_path in methods:
            print(f"\n[{task_id} / {label}] rolling out '{text}'")

            ns = SimpleNamespace(
                ckpt=str(ckpt_path), config=None,
                device=args.device, output_dir=None,
                manifest=args.manifest, data_dir=args.data_dir,
                split=args.split, T=args.T, image_size=args.image_size,
            )
            model, cfg, dev = load_model_for_eval(ns)

            sh_dim = cfg["gs_param"]["gs_dimension"] - 11
            ds, _ = build_eval_loader(
                ns, sh_dim, n_samples=args.scene_index + 1,
            )
            scene_batch = collate_batch([ds[args.scene_index]])

            with torch.no_grad():
                traj = rollout_trajectory(
                    model, text=text, scene_batch=scene_batch,
                    device=dev, enable_physics=args.enable_physics,
                )

            valid_steps = sample_timesteps(len(traj), timesteps)
            for t_target, t_actual in zip(timesteps, valid_steps):
                out_path = args.output_dir / task_id / label / f"t{t_target:03d}.png"
                render_scene_to_png(traj[t_actual], camera, out_path, image_size)
                print(f"   ✔ {out_path.relative_to(args.output_dir)}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n✓ All frames written under {args.output_dir}")
    print(f"  Now run figure_qualitative_comparison.py to compose them.")


if __name__ == "__main__":
    main()
