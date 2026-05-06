"""aggregate.py — pull eval JSONs across ablation variants → Tab 6 table.

Reads ``runs/module/<variant>/seed_<S>/eval_{a,b}/<eval>/results.json``
plus the corresponding ``runs/module/_main/seed_<S>/...`` for the
unablated baseline, and produces:

  - <out_dir>/table6.csv         — wide-format CSV
  - <out_dir>/table6.md          — Markdown table for paper draft / Notion
  - <out_dir>/raw.json           — flat dump of every (variant, dataset, metric) tuple

Conventions:
  - The "main" row uses runs/module/_main/... (auto-populated by
    eval_all.sh) — this is the SAME ckpt as runs/main_a/seed_S, just
    re-evaluated under identical eval args.
  - Missing JSONs render as ``-`` (eval likely failed).  See the per-variant
    train logs for diagnostics.

PDF mapping: §5.2 实验五 Table 1 / 消融实验; Experiment.md Tab. 6.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pull canonical variant names + descriptions from variants.py
import sys
_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))
import variants as variants_mod   # noqa: E402


# ───────────────────────────────────────────────────────────────────────
# Metric extraction map.  Each entry says: from <eval_subdir>/results.json
# pull <key path>, name it <column name>.
#
# If you add a new eval, register it here so the table picks it up.
# ───────────────────────────────────────────────────────────────────────

METRICS = [
    # (eval subdir,         json key path,        column name in table)
    ("algebraic_gaps",      "closure_mean",       "Δ_clos"),
    ("algebraic_gaps",      "inverse_mean",       "Δ_inv"),
    ("algebraic_gaps",      "commutator_mean",    "Δ_comm"),
    ("trajectory_metrics",  "ade_mean",           "ADE"),
    ("trajectory_metrics",  "fde_mean",           "FDE"),
    ("trajectory_metrics",  "mpjpe_mean",         "MPJPE"),
    ("success_rate",        "overall_mean",       "Success"),
    ("diversity",           "levenshtein_mean",   "Lev"),
    ("diversity",           "codebook_kl",        "Codebook-KL"),
]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _walk(d: Dict[str, Any], dotted: str) -> Optional[Any]:
    """Look up a.b.c in nested dict; tolerate missing keys."""
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _format_value(v: Any) -> str:
    if v is None:
        return "-"
    # Tuple = (mean, std) — used for the multi-seed main row.
    if isinstance(v, tuple) and len(v) == 2:
        m, s = v
        if m is None:    return "-"
        return f"{_fmt_num(m)} ± {_fmt_num(s)}"
    if isinstance(v, (int, float)):
        return _fmt_num(v)
    return str(v)


def _fmt_num(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.4f}"


def _aggregate_seeds(seed_dirs: List[Path]) -> Dict[str, Any]:
    """Pull metrics from each seed's eval_<ds> dir, return {col: (mean, std)}."""
    per_seed: List[Dict[str, Any]] = [collect_one(d) for d in seed_dirs if d.exists()]
    if not per_seed:
        return {col: None for _, _, col in METRICS}
    out: Dict[str, Any] = {}
    for _, _, col in METRICS:
        vals = [r[col] for r in per_seed if isinstance(r.get(col), (int, float))]
        if not vals:
            out[col] = None
        elif len(vals) == 1:
            out[col] = (vals[0], 0.0)
        else:
            mean = sum(vals) / len(vals)
            var  = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            out[col] = (mean, math.sqrt(var))
    return out


def collect_one(eval_dir: Path) -> Dict[str, Any]:
    """For a single eval root (e.g. .../seed_0/eval_a/), pull all metric values."""
    row: Dict[str, Any] = {}
    for sub, key, col in METRICS:
        j = _read_json(eval_dir / sub / "results.json")
        row[col] = _walk(j, key) if j is not None else None
    return row


def build_rows(
    run_root:    Path,
    main_root:   Path,
    seed:        int,
    variants:    List[str],
    datasets:    List[str] = ("a", "b"),
    main_seeds:  List[int] = (0, 1, 2),
) -> List[Dict[str, Any]]:
    """One row per (variant, dataset).
    Main row aggregates across ``main_seeds`` (mean ± std);
    ablation rows are single-seed.
    """
    rows: List[Dict[str, Any]] = []

    # ── Main row: aggregate over 3 seeds ────────────────────────────
    for ds in datasets:
        seed_dirs = [main_root / f"seed_{ms}" / f"eval_{ds}" for ms in main_seeds]
        existing  = [d for d in seed_dirs if d.exists()]
        if not existing:
            continue
        row = {"variant": f"main (n={len(existing)})", "dataset": ds,
               **_aggregate_seeds(existing)}
        rows.append(row)

    # ── Ablation rows: single seed ──────────────────────────────────
    for v in variants:
        for ds in datasets:
            eval_dir = run_root / v / f"seed_{seed}" / f"eval_{ds}"
            if not eval_dir.exists():
                continue
            row = {"variant": v, "dataset": ds, **collect_one(eval_dir)}
            rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, Any]], out: Path) -> None:
    if not rows:
        return
    cols = ["variant", "dataset"] + [m[2] for m in METRICS]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([_format_value(r.get(c)) for c in cols])


def write_md(rows: List[Dict[str, Any]], out: Path) -> None:
    cols = ["variant", "dataset"] + [m[2] for m in METRICS]
    lines = [
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(_format_value(r.get(c)) for c in cols) + " |")

    with open(out, "w") as f:
        f.write("# Table 6 — Module ablation\n\n")
        f.write("Generated by `eval/ablation/module/aggregate.py`. ")
        f.write("Numbers are means over the eval batches; "
                "`-` indicates the corresponding eval JSON was missing or unparseable.\n\n")
        f.write("\n".join(lines) + "\n\n")
        f.write("## Variants\n\n")
        for v in variants_mod.list_variants():
            d = variants_mod.get_variant(v)["description"].splitlines()[0]
            f.write(f"- **{v}** — {d}\n")


def write_raw(rows: List[Dict[str, Any]], out: Path) -> None:
    with open(out, "w") as f:
        json.dump(rows, f, indent=2, default=str)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-root",  type=str, default="runs/module")
    p.add_argument("--main-root", type=str, default="runs/module/_main")
    p.add_argument("--seed",      type=int, default=0)
    p.add_argument("--variants",  nargs="+", default=variants_mod.list_variants())
    p.add_argument("--out-dir",   type=str, default="runs/module/_aggregate")
    p.add_argument("--datasets",  nargs="+", default=["a"], choices=["a", "b"],
                   help="Which dataset halves to include (default: A only). "
                        "Pass --datasets a b if you've also run B fine-tunes.")
    p.add_argument("--main-seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="Seeds of the pre-trained main model to average for "
                        "the baseline row (default: 0 1 2).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(
        run_root   = Path(args.run_root),
        main_root  = Path(args.main_root),
        seed       = args.seed,
        variants   = args.variants,
        datasets   = args.datasets,
        main_seeds = args.main_seeds,
    )
    write_csv(rows, out_dir / "table6.csv")
    write_md(rows,  out_dir / "table6.md")
    write_raw(rows, out_dir / "raw.json")

    print(f"  ✔ wrote {out_dir / 'table6.csv'}")
    print(f"  ✔ wrote {out_dir / 'table6.md'}")
    print(f"  ✔ wrote {out_dir / 'raw.json'}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
