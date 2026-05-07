"""extract_val_curves.py — Plot per-variant val_loss curves overlaid.

Parses ``runs/{module,loss}/<variant>/seed_0/train_a.log`` (and main_a logs)
to extract per-epoch val_loss values across all 4 curriculum stages, then
plots them on a single figure.

Output:
  <out_dir>/val_curves.png   ← Paper Fig 4 candidate (NeurIPS 2-column wide)
  <out_dir>/val_curves.csv   ← Raw data, per-row (variant, epoch, stage, val, best)
  <out_dir>/val_curves.md    ← Quick text summary of best vals + plateaus

Usage::
    python eval/ablation/analysis/extract_val_curves.py
        --out-dir runs/_analysis
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Patterns we extract from train logs ────────────────────────────────
STAGE_RE = re.compile(r"=== Stage (\w+) ===")
EPOCH_DONE_RE = re.compile(r"\s*epoch\s+(\d+)/(\d+)\s+done in\s+([\d.]+)s\s+step=(\d+)")
VAL_RE = re.compile(r"\s*val_loss=([\d.eE+-]+|nan)\s+\(best=([\d.eE+-]+|nan)\)")


def parse_log(log_path: Path) -> List[Dict]:
    """Return list of {cum_ep, stage, stage_ep, val, best} per epoch."""
    if not log_path.exists():
        return []
    rows: List[Dict] = []
    cur_stage: Optional[str] = None
    cur_stage_ep: Optional[int] = None
    cum_ep = 0

    with open(log_path) as f:
        for line in f:
            m = STAGE_RE.search(line)
            if m:
                cur_stage = m.group(1)
                cur_stage_ep = None
                continue
            m = EPOCH_DONE_RE.match(line)
            if m:
                cur_stage_ep = int(m.group(1))
                continue
            m = VAL_RE.match(line)
            if m and cur_stage and cur_stage_ep is not None:
                cum_ep += 1
                try:
                    val = float(m.group(1))
                    best = float(m.group(2))
                except ValueError:
                    continue
                rows.append({
                    "cum_ep":   cum_ep,
                    "stage":    cur_stage,
                    "stage_ep": cur_stage_ep,
                    "val":      val,
                    "best":     best,
                })
    return rows


def collect_variants(run_root: Path) -> Dict[str, Path]:
    """Auto-discover variants under run_root/<variant>/seed_0/train_a.log."""
    out: Dict[str, Path] = {}
    if not run_root.exists():
        return out
    for variant_dir in sorted(run_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        log = variant_dir / "seed_0" / "train_a.log"
        if log.exists():
            out[variant_dir.name] = log
    return out


def collect_main(main_seeds: List[int] = (0, 1, 2)) -> Dict[str, Path]:
    """Find the main_a logs (per-seed)."""
    out = {}
    for s in main_seeds:
        for log_name in ("train.log", "train_a.log"):
            for sub in (Path("runs/main_a"), Path("runs/main_exp")):
                p = sub / f"seed_{s}" / log_name
                if p.exists():
                    out[f"main_seed{s}"] = p
                    break
    return out


# ── Plotting ────────────────────────────────────────────────────────────
PALETTE = {
    "main_seed0":      "#000000",
    "main_seed1":      "#444444",
    "main_seed2":      "#888888",
    # module
    "no_algebraic":    "#1f77b4",
    "no_physics":      "#d62728",
    "no_hier":         "#2ca02c",
    "no_cvae":         "#ff7f0e",
    "no_equivariance": "#9467bd",
    "no_lipschitz":    "#8c564b",
    # loss
    "no_L_clos":       "#17becf",
    "no_L_inv":        "#e377c2",
    "no_L_eq":         "#bcbd22",
    "no_L_hier":       "#7f7f7f",
    "no_L_nce":        "#1abc9c",
    "no_L_comm":       "#34495e",
    "no_kl_anneal":    "#e67e22",
}

STAGE_BOUNDARIES = [25, 45, 60, 95]  # cum_ep where each stage ends (75-ep budget)


def plot_curves(
    variants_data: Dict[str, List[Dict]],
    out_path: Path,
    title: str = "Validation loss across ablation variants",
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    for name in sorted(variants_data, key=lambda x: (not x.startswith("main"), x)):
        rows = variants_data[name]
        if not rows:
            continue
        ep = [r["cum_ep"] for r in rows]
        val = [r["val"] for r in rows]
        color = PALETTE.get(name, None)
        is_main = name.startswith("main")
        ax.plot(
            ep, val,
            color=color, label=name,
            linewidth=1.6 if is_main else 1.1,
            linestyle="-" if not is_main else "--",
            alpha=0.95 if is_main else 0.85,
        )

    # Stage shading
    for x in STAGE_BOUNDARIES[:-1]:
        ax.axvline(x, color="grey", linestyle=":", linewidth=0.6, alpha=0.5)
    stage_labels = ["RIGID", "PLANNER", "PHYSICS", "FULL"]
    stage_centers = [12.5, 35, 52.5, 77.5]
    for label, x in zip(stage_labels, stage_centers):
        ax.text(x, ax.get_ylim()[1] * 0.97, label,
                ha="center", va="top", fontsize=9,
                color="grey", style="italic")

    ax.set_xlabel("Cumulative epoch", fontsize=11)
    ax.set_ylabel("Validation loss (lower is better)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, framealpha=0.92, ncol=1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"  ✔ saved figure: {out_path}")


def write_csv(variants_data: Dict[str, List[Dict]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "cum_ep", "stage", "stage_ep", "val_loss", "best_val"])
        for name, rows in variants_data.items():
            for r in rows:
                w.writerow([name, r["cum_ep"], r["stage"], r["stage_ep"],
                            r["val"], r["best"]])
    print(f"  ✔ saved data: {out_path}")


def write_md(variants_data: Dict[str, List[Dict]], out_path: Path) -> None:
    """Short text summary: best val per variant + final val per variant."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Validation curves — quick summary\n"]
    lines.append("| variant | epochs logged | best val_loss | final val_loss | Δ vs final |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(variants_data):
        rows = variants_data[name]
        if not rows:
            continue
        best = min(r["best"] for r in rows)
        final = rows[-1]["val"]
        lines.append(f"| {name} | {len(rows)} | {best:.4f} | {final:.4f} | "
                     f"{final - best:+.4f} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  ✔ saved summary: {out_path}")


# ── Main ────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--module-root", type=str, default="runs/module")
    p.add_argument("--loss-root",   type=str, default="runs/loss")
    p.add_argument("--main-seeds",  nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir",     type=str, default="runs/_analysis")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    print("=== Discovering variants ===")
    sources = {}
    sources.update(collect_main(args.main_seeds))
    sources.update(collect_variants(Path(args.module_root)))
    sources.update(collect_variants(Path(args.loss_root)))
    print(f"  found {len(sources)} log(s)")

    print("\n=== Parsing logs ===")
    data = {}
    for name, log in sorted(sources.items()):
        rows = parse_log(log)
        data[name] = rows
        print(f"  {name:20s}  {len(rows):>4d} epochs from {log}")

    print("\n=== Writing outputs ===")
    plot_curves(data, out_dir / "val_curves.png")
    write_csv(data, out_dir / "val_curves.csv")
    write_md(data,  out_dir / "val_curves.md")
    print(f"\n  done → {out_dir}/")


if __name__ == "__main__":
    main()
