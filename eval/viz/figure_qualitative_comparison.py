"""
figure_qualitative_comparison.py — NeurIPS main-paper qualitative figure.

Produces a side-by-side comparison grid:
  - rows: methods (Ours + 3-4 baselines)
  - columns: timesteps (e.g., t=0, 2, 4, 6, 8, 10)
  - groups: tasks (e.g., "open drawer (unseen)", "5-step long-horizon")

Visual elements:
  - "Ours" row highlighted with light green tint
  - Failed frames marked with red border + optional text annotation
  - Success/failure indicator (✓/✗) at right of each row
  - Task title above each task block
  - Camera-consistent rendering across methods (assumed pre-rendered)

Workflow
--------
1. For each method, render its rollout from a fixed camera viewpoint
   into PNG frames (one per timestep).  Place them under:
       <frames_dir>/<task>/<method>/t000.png, t002.png, ...
2. Define which frames are "failed" via the failures.json config.
3. Run this script to compose the grid into a single PDF/PNG.

Example::
    python -m eval.viz.figure_qualitative_comparison \\
        --frames-dir runs/figures/qualitative/frames \\
        --tasks open_drawer_unseen long_horizon_5step \\
        --methods Ours PhysGaussian SVD MotionGPT \\
        --timesteps 0 2 4 6 8 10 \\
        --failures runs/figures/qualitative/failures.json \\
        --output runs/figures/qualitative/fig_qualitative.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# Default styling (publication-friendly)
# ──────────────────────────────────────────────────────────────────────

OURS_NAME = "Ours"
OURS_TINT = (0.85, 1.00, 0.85)        # light green background for "Ours" row
FAIL_COLOR = "#D62728"                # red for failed-frame border
SUCCESS_COLOR = "#2CA02C"             # green for ✓
FAIL_INDICATOR = "#D62728"            # red for ✗
PARTIAL_COLOR = "#FF7F0E"             # orange for ⚠

PLACEHOLDER_BG = (0.92, 0.92, 0.92)   # light grey placeholder if frame missing


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────

class FailureSpec:
    """Tracks which (task, method, timestep) cells are failures.

    Loaded from a JSON file with structure::

        {
          "open_drawer_unseen": {
            "Ours":          { "result": "success", "failed_steps": [] },
            "PhysGaussian":  { "result": "fail",    "failed_steps": [4, 6, 8, 10],
                               "annotation": "Object distorted at t=4" },
            "SVD":           { "result": "partial", "failed_steps": [8, 10] }
          },
          "long_horizon_5step": { ... }
        }
    """

    def __init__(self, spec: Dict[str, Dict[str, dict]]):
        self.spec = spec

    @classmethod
    def load(cls, path: Optional[Path]) -> "FailureSpec":
        if path is None or not Path(path).exists():
            return cls({})
        with open(path) as f:
            return cls(json.load(f))

    def is_failed(self, task: str, method: str, t: int) -> bool:
        entry = self.spec.get(task, {}).get(method, {})
        return t in entry.get("failed_steps", [])

    def result(self, task: str, method: str) -> str:
        """Return 'success' / 'fail' / 'partial' / 'unknown'."""
        return self.spec.get(task, {}).get(method, {}).get("result", "unknown")

    def annotation(self, task: str, method: str) -> Optional[str]:
        return self.spec.get(task, {}).get(method, {}).get("annotation")


# ──────────────────────────────────────────────────────────────────────
# Frame loading
# ──────────────────────────────────────────────────────────────────────

def _load_frame(path: Path, fallback_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load a PNG frame; return a placeholder grey image if the file is missing."""
    if path.exists():
        img = Image.open(path).convert("RGB")
        return np.asarray(img)
    H, W = fallback_size
    arr = np.ones((H, W, 3), dtype=np.float32)
    arr[..., 0] *= PLACEHOLDER_BG[0]
    arr[..., 1] *= PLACEHOLDER_BG[1]
    arr[..., 2] *= PLACEHOLDER_BG[2]
    return (arr * 255).astype(np.uint8)


def _frame_path(frames_dir: Path, task: str, method: str, t: int) -> Path:
    return frames_dir / task / method / f"t{t:03d}.png"


# ──────────────────────────────────────────────────────────────────────
# Single task block rendering
# ──────────────────────────────────────────────────────────────────────

def _draw_task_block(
    fig: plt.Figure,
    gs_block: matplotlib.gridspec.SubplotSpec,
    *,
    task: str,
    task_title: str,
    methods: List[str],
    timesteps: List[int],
    frames_dir: Path,
    failures: FailureSpec,
    show_time_labels: bool,
) -> None:
    """Render one task block: method rows × timestep columns + side indicator."""
    n_methods = len(methods)
    n_steps = len(timesteps)

    # Inner grid: title row + time-label row + n_methods rows of [label | T cells | indicator]
    inner = gs_block.subgridspec(
        n_methods + 2, n_steps + 2,
        width_ratios=[1.4] + [1.0] * n_steps + [0.5],
        height_ratios=[0.30, 0.18] + [1.0] * n_methods,
        wspace=0.05, hspace=0.08,
    )

    # Task title spans the full width
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5, 0.5, task_title,
        ha="center", va="center",
        fontsize=11, fontweight="bold",
        transform=title_ax.transAxes,
    )

    # Time labels along the second row
    if show_time_labels:
        for j, t in enumerate(timesteps):
            tax = fig.add_subplot(inner[1, 1 + j])
            tax.axis("off")
            tax.text(
                0.5, 0.2, f"t={t}",
                ha="center", va="bottom",
                fontsize=9, color="#333333",
                transform=tax.transAxes,
            )

    # Method rows (offset by +2 since rows 0 and 1 are title and time labels)
    for r, method in enumerate(methods):
        is_ours = (method == OURS_NAME)

        # Method label
        label_ax = fig.add_subplot(inner[r + 2, 0])
        label_ax.axis("off")
        if is_ours:
            label_ax.add_patch(patches.Rectangle(
                (0, 0), 1, 1, transform=label_ax.transAxes,
                facecolor=OURS_TINT, edgecolor="none", zorder=0,
            ))
        label_ax.text(
            0.5, 0.5, method,
            ha="center", va="center",
            fontsize=10, fontweight="bold" if is_ours else "normal",
            transform=label_ax.transAxes,
            zorder=1,
        )

        # Timestep cells
        for c, t in enumerate(timesteps):
            cell_ax = fig.add_subplot(inner[r + 2, 1 + c])
            img = _load_frame(_frame_path(frames_dir, task, method, t))
            cell_ax.imshow(img)
            cell_ax.set_xticks([])
            cell_ax.set_yticks([])
            for spine in cell_ax.spines.values():
                spine.set_visible(False)

            # Light green tint background for Ours row
            if is_ours:
                cell_ax.add_patch(patches.Rectangle(
                    (0, 0), 1, 1, transform=cell_ax.transAxes,
                    facecolor=OURS_TINT, edgecolor="none", zorder=-1,
                ))

            # Red failure border
            if failures.is_failed(task, method, t):
                cell_ax.add_patch(patches.Rectangle(
                    (0.005, 0.005), 0.99, 0.99,
                    transform=cell_ax.transAxes,
                    linewidth=2.5, edgecolor=FAIL_COLOR,
                    facecolor="none", zorder=2,
                ))

        # Result indicator (✓ / ✗ / ⚠) at right
        ind_ax = fig.add_subplot(inner[r + 2, -1])
        ind_ax.axis("off")
        result = failures.result(task, method)
        if result == "success":
            ind_ax.text(0.5, 0.5, "✓",  # ✓
                        ha="center", va="center", fontsize=22,
                        color=SUCCESS_COLOR, fontweight="bold",
                        transform=ind_ax.transAxes)
        elif result == "fail":
            ind_ax.text(0.5, 0.5, "✗",  # ✗
                        ha="center", va="center", fontsize=22,
                        color=FAIL_INDICATOR, fontweight="bold",
                        transform=ind_ax.transAxes)
        elif result == "partial":
            ind_ax.text(0.5, 0.5, "⚠",  # ⚠
                        ha="center", va="center", fontsize=20,
                        color=PARTIAL_COLOR, fontweight="bold",
                        transform=ind_ax.transAxes)
        # else: unknown → no marker

        # Optional annotation under the indicator
        annotation = failures.annotation(task, method)
        if annotation:
            ind_ax.text(0.5, 0.05, annotation,
                        ha="center", va="bottom",
                        fontsize=6, color="#555555",
                        transform=ind_ax.transAxes,
                        wrap=True)


# ──────────────────────────────────────────────────────────────────────
# Top-level figure assembly
# ──────────────────────────────────────────────────────────────────────

def make_figure(
    frames_dir: Path,
    tasks: List[Tuple[str, str]],          # list of (task_id, task_title)
    methods: List[str],
    timesteps: List[int],
    failures: FailureSpec,
    output_path: Path,
    cell_inches: float = 0.9,
    dpi: int = 200,
) -> None:
    """Assemble the full figure of multiple task blocks, save to disk."""
    n_tasks = len(tasks)
    n_methods = len(methods)
    n_steps = len(timesteps)

    # Per-block height: title row (0.30) + time-label row (0.18) + n_methods cell rows
    block_h = (0.30 + 0.18 + n_methods) * cell_inches + 0.3
    fig_h = block_h * n_tasks
    # Width: label col (1.4) + n_steps cell cols (1.0) + indicator col (0.5)
    fig_w = (1.4 + n_steps + 0.5) * cell_inches

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    outer_gs = fig.add_gridspec(
        n_tasks, 1, hspace=0.35,
    )

    for ti, (task_id, task_title) in enumerate(tasks):
        _draw_task_block(
            fig, outer_gs[ti, 0],
            task=task_id,
            task_title=f"({chr(ord('a') + ti)}) {task_title}",
            methods=methods,
            timesteps=timesteps,
            frames_dir=frames_dir,
            failures=failures,
            show_time_labels=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved figure to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Demo data generator (for testing the layout without real renders)
# ──────────────────────────────────────────────────────────────────────

def _generate_demo_frames(
    frames_dir: Path,
    tasks: List[str],
    methods: List[str],
    timesteps: List[int],
    failures: FailureSpec,
    image_size: Tuple[int, int] = (256, 256),
) -> None:
    """Create dummy PNG frames so you can preview the layout."""
    import matplotlib.pyplot as plt
    H, W = image_size
    rng = np.random.default_rng(0)

    for task in tasks:
        for method in methods:
            for t in timesteps:
                p = _frame_path(frames_dir, task, method, t)
                p.parent.mkdir(parents=True, exist_ok=True)

                fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.set_xticks([]); ax.set_yticks([])

                # Draw a synthetic "object" that progresses with t (and breaks for failed methods)
                progress = t / max(timesteps)
                if failures.is_failed(task, method, t):
                    # Break visually: scrambled positions
                    pts = rng.uniform(0, 1, size=(80, 2))
                    color = "#A00000"
                else:
                    # Smoothly opening "drawer" or moving object
                    pts = np.column_stack([
                        np.linspace(0.2, 0.5 + 0.3 * progress, 80),
                        np.linspace(0.4, 0.4, 80) + rng.normal(0, 0.02, 80),
                    ])
                    color = "#3070C0" if method == OURS_NAME else "#606060"
                ax.scatter(pts[:, 0], pts[:, 1], s=14, c=color, alpha=0.7)
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                fig.savefig(p, dpi=100)
                plt.close(fig)
    print(f"✓ Generated demo frames under {frames_dir}")


def _default_failure_spec(tasks: List[str], methods: List[str]) -> FailureSpec:
    """A reasonable default for demo: Ours succeeds, baselines fail at later steps."""
    spec: Dict[str, Dict[str, dict]] = {}
    for task in tasks:
        spec[task] = {}
        for m in methods:
            if m == OURS_NAME:
                spec[task][m] = {"result": "success", "failed_steps": []}
            elif m == "PhysGaussian":
                spec[task][m] = {
                    "result": "fail",
                    "failed_steps": [4, 6, 8, 10],
                    "annotation": "Distorted at t=4",
                }
            elif m == "SVD":
                spec[task][m] = {
                    "result": "partial",
                    "failed_steps": [8, 10],
                    "annotation": "Drift at t=8",
                }
            elif m == "MotionGPT":
                spec[task][m] = {
                    "result": "fail",
                    "failed_steps": [4, 6, 8, 10],
                    "annotation": "Joint-space jump",
                }
            elif m == "TAMP-PDDL":
                spec[task][m] = {
                    "result": "fail",
                    "failed_steps": [6, 8, 10],
                    "annotation": "Plan failed",
                }
            else:
                spec[task][m] = {"result": "unknown", "failed_steps": []}
    return FailureSpec(spec)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _parse_tasks(specs: List[str]) -> List[Tuple[str, str]]:
    """Parse 'task_id:Display Title' or just 'task_id' (uses id as title)."""
    out: List[Tuple[str, str]] = []
    for s in specs:
        if ":" in s:
            tid, title = s.split(":", 1)
            out.append((tid.strip(), title.strip()))
        else:
            out.append((s.strip(), s.strip().replace("_", " ")))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", type=Path, required=True,
                   help="Root directory of pre-rendered PNG frames "
                        "(<dir>/<task>/<method>/t000.png ...)")
    p.add_argument("--tasks", nargs="+", required=True,
                   help="Task ids; optionally 'task_id:Display Title'.")
    p.add_argument("--methods", nargs="+", required=True,
                   help="Method names. 'Ours' will be highlighted automatically.")
    p.add_argument("--timesteps", nargs="+", type=int, default=[0, 2, 4, 6, 8, 10],
                   help="Timestep indices to display per row.")
    p.add_argument("--failures", type=Path, default=None,
                   help="Optional JSON describing failures per (task, method).")
    p.add_argument("--output", type=Path,
                   default=Path("runs/figures/qualitative/fig_qualitative.pdf"))
    p.add_argument("--cell-inches", type=float, default=0.9,
                   help="Per-cell width/height in inches.")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--demo", action="store_true",
                   help="Generate dummy demo frames + default failure spec to "
                        "preview the layout without real renders.")
    args = p.parse_args()

    tasks = _parse_tasks(args.tasks)
    task_ids = [t[0] for t in tasks]

    if args.demo:
        # Use a default failure spec if --failures not provided
        if args.failures is None:
            failures = _default_failure_spec(task_ids, args.methods)
            print("(demo) Using default failure spec.")
        else:
            failures = FailureSpec.load(args.failures)
        _generate_demo_frames(args.frames_dir, task_ids, args.methods,
                              args.timesteps, failures)
    else:
        failures = FailureSpec.load(args.failures)

    make_figure(
        frames_dir=args.frames_dir,
        tasks=tasks,
        methods=args.methods,
        timesteps=args.timesteps,
        failures=failures,
        output_path=args.output,
        cell_inches=args.cell_inches,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
