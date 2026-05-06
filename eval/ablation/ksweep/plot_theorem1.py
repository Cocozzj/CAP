"""plot_theorem1.py — Empirical verification plot for Theorem 1.

Read the summary.json produced by ``eval_sweep.sh`` (which calls
``eval.k_scaling_sweep``) and render a NeurIPS-ready figure showing:

  - measured closure / inverse / commutator gaps vs codebook size K
  - theoretical envelope ``A · K^(-1/d)`` overlaid (using fitted A, d)
  - log-log axes (the relationship is a straight line in log-log space)
  - title / legend / axis labels camera-ready

Usage::

    python eval/ablation/ksweep/plot_theorem1.py \\
        --summary runs/ablation/ksweep/_eval/summary.json \\
        --output  runs/ablation/ksweep/_eval/theorem1_plot.pdf

Optional: ``--metrics closure inverse`` to only plot a subset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless servers
import matplotlib.pyplot as plt


METRIC_LABELS = {
    "closure":    r"Closure  $\Delta_{\mathrm{clos}}$",
    "inverse":    r"Inverse  $\Delta_{\mathrm{inv}}$",
    "commutator": r"Commutator  $\Delta_{\mathrm{comm}}$",
}
METRIC_COLORS = {
    "closure":    "#1f77b4",
    "inverse":    "#d62728",
    "commutator": "#2ca02c",
}


def _load(summary_path: Path) -> Dict:
    with open(summary_path) as f:
        return json.load(f)


def _plot_metric(ax, K_arr: np.ndarray, e_arr: np.ndarray,
                 fit: Dict, name: str, color: str) -> None:
    """Plot one metric: scatter + fitted theoretical line."""
    # Scatter measured points.
    ax.loglog(K_arr, e_arr, "o", color=color, markersize=8,
              markeredgecolor="black", markeredgewidth=0.6,
              label=f"{METRIC_LABELS.get(name, name)}  (measured)")

    # Theoretical line  err = A · K^(-1/d) .
    if fit and not np.isnan(fit.get("A", np.nan)) and not np.isnan(fit.get("d", np.nan)):
        K_smooth = np.geomspace(K_arr.min() * 0.7, K_arr.max() * 1.4, 100)
        e_pred = fit["A"] * np.power(K_smooth, -1.0 / fit["d"])
        ax.loglog(K_smooth, e_pred, "--", color=color, linewidth=1.4, alpha=0.85,
                  label=(f"  fit:  $A K^{{-1/d}}$,  "
                         f"$d={fit['d']:.2f}$,  $R^2={fit['r2']:.2f}$"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=str, required=True,
                   help="summary.json from eval.k_scaling_sweep")
    p.add_argument("--output", type=str, required=True,
                   help="Output figure path (.pdf or .png)")
    p.add_argument("--metrics", nargs="+",
                   default=["closure", "inverse", "commutator"],
                   choices=list(METRIC_LABELS.keys()))
    p.add_argument("--title", type=str,
                   default=r"Theorem 1: closure error scales as $K^{-1/d}$")
    p.add_argument("--width",  type=float, default=6.5,   # 1-column NeurIPS
                   help="Figure width in inches (default 6.5)")
    p.add_argument("--height", type=float, default=4.2)
    args = p.parse_args()

    summary = _load(Path(args.summary))
    Ks = sorted(int(k) for k in summary["per_K"].keys())
    fits = summary.get("fits", {})

    fig, ax = plt.subplots(figsize=(args.width, args.height))

    for m in args.metrics:
        e_arr = np.array([summary["per_K"][str(k)][m] for k in Ks], dtype=float)
        K_arr = np.array(Ks, dtype=float)

        # Skip non-positive entries (log undefined).
        mask = (e_arr > 0) & np.isfinite(e_arr)
        if mask.sum() < 2:
            print(f"  skip {m!r}: < 2 valid points")
            continue
        _plot_metric(ax, K_arr[mask], e_arr[mask],
                     fits.get(m, {}), name=m, color=METRIC_COLORS[m])

    ax.set_xlabel(r"Atomic codebook size $K$", fontsize=11)
    ax.set_ylabel(r"Algebraic gap (mean over test batches)", fontsize=11)
    ax.set_title(args.title, fontsize=12)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.legend(loc="best", fontsize=9, framealpha=0.92)

    # Tick formatting: explicit K values on x-axis.
    ax.set_xticks(Ks)
    ax.set_xticklabels([str(k) for k in Ks])
    ax.tick_params(axis="both", labelsize=9)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight", dpi=200)
    print(f"  ✔ saved figure: {out}")

    # Also save a small text summary alongside.
    txt = out.with_suffix(".txt")
    with open(txt, "w") as f:
        f.write(f"K values: {Ks}\n\n")
        for m in args.metrics:
            fit = fits.get(m, {})
            f.write(f"[{m}]\n")
            for k in Ks:
                v = summary["per_K"][str(k)].get(m, float("nan"))
                f.write(f"  K={k:>5d}  err={v:.5f}\n")
            if fit:
                f.write(f"  fit: A={fit.get('A', float('nan')):.4f}  "
                        f"d={fit.get('d', float('nan')):.3f}  "
                        f"R²={fit.get('r2', float('nan')):.3f}  "
                        f"(n={fit.get('n', 0)})\n")
            f.write("\n")
    print(f"  ✔ saved summary: {txt}")


if __name__ == "__main__":
    main()
