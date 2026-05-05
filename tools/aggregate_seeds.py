"""
aggregate_seeds.py — pool eval results across the 5-seed sweep.

Reads ``<base>/seed_<S>/eval/<eval-name>/summary.json`` for every seed
present, computes mean ± std for each scalar field, and writes a single
``<base>/agg/<eval-name>.json`` plus a markdown table for the paper.

Usage::

    python tools/aggregate_seeds.py runs/main_exp --eval-name algebraic_gaps
    python tools/aggregate_seeds.py runs/main_exp --eval-name success_rate

Handles two summary shapes uniformly:
  - flat:   {metric: value, ...}
  - nested: {metric: {mean, std, n}, ...}      (algebraic_gaps style)
  - per-task: {per_task: {task: {metric: ...}}, overall_*: ...}

Skips fields that are not numeric.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _flatten(d: Any, prefix: str = "") -> Dict[str, float]:
    """Walk a nested dict, emit ``flat.path: scalar`` for every numeric leaf.

    Treats ``{"mean": x, "std": y}`` blocks as a single leaf with the mean —
    we re-aggregate ourselves across seeds.
    """
    out: Dict[str, float] = {}
    if isinstance(d, (int, float)) and not isinstance(d, bool):
        out[prefix.rstrip(".")] = float(d)
        return out
    if not isinstance(d, dict):
        return out
    if "mean" in d and isinstance(d["mean"], (int, float)):
        out[prefix.rstrip(".")] = float(d["mean"])
        return out
    for k, v in d.items():
        out.update(_flatten(v, prefix=f"{prefix}{k}."))
    return out


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))],
                     dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std(ddof=0)),
        "n":    int(arr.size),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
    }


def _md_table(rows: List[Tuple[str, Dict[str, float]]]) -> str:
    """Render seed-aggregated results as a paper-ready markdown table."""
    lines = ["| Metric | Mean ± Std (n) | Min | Max |",
             "|---|---|---|---|"]
    for name, s in rows:
        if s["n"] == 0:
            lines.append(f"| `{name}` | n/a | — | — |")
        else:
            lines.append(
                f"| `{name}` | {s['mean']:.4f} ± {s['std']:.4f} ({s['n']}) | "
                f"{s['min']:.4f} | {s['max']:.4f} |"
            )
    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("base", type=str,
                   help="Base run dir, e.g. runs/main_exp (contains seed_<S>/)")
    p.add_argument("--eval-name", type=str, required=True,
                   help="Subdirectory under <seed>/eval/, e.g. algebraic_gaps")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where to write aggregate (default: <base>/agg/)")
    args = p.parse_args()

    base = Path(args.base)
    out  = Path(args.out_dir) if args.out_dir else base / "agg"
    out.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(base.glob("seed_*"))
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {base}")

    # ── Collect flattened metrics per seed ──
    per_seed: Dict[int, Dict[str, float]] = {}
    for sd in seed_dirs:
        try:
            seed = int(sd.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        summary_path = sd / "eval" / args.eval_name / "summary.json"
        if not summary_path.exists():
            print(f"  - skip seed={seed}: no {summary_path.relative_to(base)}")
            continue
        with open(summary_path) as f:
            summary = json.load(f)
        per_seed[seed] = _flatten(summary)
        print(f"  ✓ seed={seed}: {len(per_seed[seed])} metrics")

    if not per_seed:
        raise SystemExit(f"No summary.json found for eval-name={args.eval_name}")

    # ── Pool by metric name ──
    metric_names = sorted({m for d in per_seed.values() for m in d})
    aggregated: Dict[str, Dict[str, float]] = {}
    for m in metric_names:
        aggregated[m] = _stats([d[m] for d in per_seed.values() if m in d])

    # ── Persist JSON + Markdown ──
    json_path = out / f"{args.eval_name}.json"
    md_path   = out / f"{args.eval_name}.md"
    with open(json_path, "w") as f:
        json.dump({
            "eval_name":  args.eval_name,
            "base":       str(base),
            "seeds":      sorted(per_seed.keys()),
            "n_seeds":    len(per_seed),
            "per_seed":   per_seed,
            "aggregated": aggregated,
        }, f, indent=2)
    md_path.write_text(
        f"# {args.eval_name} — {len(per_seed)} seeds: {sorted(per_seed.keys())}\n\n"
        + _md_table([(m, aggregated[m]) for m in metric_names])
    )

    print(f"\n✓ Aggregated {len(metric_names)} metrics across {len(per_seed)} seeds")
    print(f"  JSON: {json_path}")
    print(f"  MD  : {md_path}")


if __name__ == "__main__":
    main()
