"""Generate paper-ready LaTeX tables from the aggregator's main_table.json.

Two output layouts:

  --summary  Wide layout (one row per baseline, columns grouped by experiment).
             This is the paper's main Table 1 — assembles 7 experiments
             (IID, unseen_pair, unseen_object, unseen_action, cross-material,
             long-horizon, diversity) into a single comparison table.

  default    Per-(dataset, split) layout (one row per baseline-per-split,
             one table per split).  Useful for diagnostics / supplementary.

Usage:

    # Main paper table (one wide summary table)
    python -m eval.baseline.format_latex \\
        --json runs/main_table.json \\
        --dataset dataset_a \\
        --summary \\
        --output runs/table_main_dataset_a.tex

    # Per-split detailed tables (one table per test split)
    python -m eval.baseline.format_latex \\
        --json runs/main_table.json \\
        --dataset dataset_a \\
        --output runs/table_per_split_dataset_a.tex
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════
# Display name mapping
# ══════════════════════════════════════════════════════════════════════

_PAPER_NAME = {
    # PDDLStream + FastDownward couldn't be built on the eval cluster
    # (CUDA-12 toolchain incompatibility with the old C++ planner).  We
    # fall back to a simpler symbolic decomposer (read action verb from
    # meta.json → look up motion primitive → emit [T,7] pose trajectory),
    # which is fully fair (every baseline sees the same GT meta.json
    # input) but isn't the full TAMP search loop.  The column header is
    # renamed to reflect this honestly; a footnote in the paper explains.
    "tamp_pddl":    "Symbolic + Motion Primitives",
    "physgaussian": "PhysGaussian",
    # Generic image-to-video diffusion prior (Blattmann et al. 2023),
    # zero-shot — pixel output only, no 3D, so 3D metric columns render
    # as "—" for this row.  See eval/baseline/svd/README.md.  PhysDreamer
    # is not included; its per-scene optimization at ~30 min/traj is
    # infeasible at our 1300+ trajectory evaluation scale.
    "svd":          "Stable Video Diffusion",
    "motiongpt":    "MotionGPT",
    "ours":         "\\textbf{Ours}",
    # Backward-compat names (only render if data exists; default row order
    # excludes them).  Useful for ablation/supplementary tables that may
    # still reference the old/legacy baselines.
    "magvit_v2":    "MAGVIT-v2 (excluded)",
    "flat_vqvae":   "Flat VQ-VAE",
    "tamp_rule":    "TAMP-rule (legacy)",
    "_4dgs":        "4D-GS (legacy)",
}

# Row order for the new 5-baseline matrix (Ours always last with \midrule).
_ROW_ORDER = [
    "tamp_pddl",
    "physgaussian",
    "physdreamer",
    "magvit_v2",
    "motiongpt",
    "ours",
]

# Per-baseline cells deliberately marked as "—" (the metric is not measurable
# for that baseline — e.g. closure_gap requires our codebook structure;
# cross-friction success requires a physics-aware predictor).
#
# Each entry can be either:
#   - a bare ``field`` string (applies to every split for that field), or
#   - a ``(split, field)`` tuple (applies only to that specific cell).
#
# Cross-Topology note: for the ``test_soft`` Succ column, baselines without a
# soft-body / deformation module (TAMP-rule, Flat VQ-VAE) are marked N/A —
# TAMP's hand-coded rules are written for articulated rigid joints and don't
# apply to deformable cloth/soft objects.  Flat VQ-VAE has no physics
# component to deform.  PhysGaussian has soft-body simulation (MPM) so it
# IS evaluable on soft objects.
_NA_OVERRIDES = {
    # TAMP (PDDLStream): symbolic + IK planner, deterministic given GT URDF.
    # No diversity.  No algebraic-structure metrics (Closure / Inverse) since
    # actions are discrete plans, not group elements in our codebook.
    # Cross-material: PDDL has no material reasoning → N/A on iron/foam.
    "tamp_pddl":    {
        "closure_gap", "inverse_gap",
        "action_diversity", "result_diversity",
        ("test_iron", "success"),
        ("test_foam", "success"),
    },
    # PhysGaussian: deterministic MPM simulator with full material support.
    # Strong on physics consistency; no diversity / no algebraic.
    "physgaussian": {
        "closure_gap", "inverse_gap",
        "action_diversity", "result_diversity",
    },
    # PhysDreamer: learned 4D physics generator (video diffusion + physics).
    # Stochastic (diversity OK).  No algebraic structure.
    "physdreamer":  {
        "closure_gap", "inverse_gap",
    },
    # MAGVIT-v2: pixel-only video tokenizer.  No 3D / no physics / no algebra.
    "magvit_v2":    {
        "ade", "fde", "mpjpe", "closure_gap", "inverse_gap",
        "phys_wasserstein", "energy_violation",
        "contact_violation", "volume_violation",
        ("test_iron",  "success"),
        ("test_foam",  "success"),
        ("test_heavy", "success"),
        ("test_light", "success"),
        # action_diversity OK (MAGVIT samples token sequences)
    },
    # MotionGPT: pretrained T5 + motion VQ.  Sampling-capable.  No physics
    # module / no algebraic structure.  Cross-material → N/A.
    "motiongpt":    {
        "closure_gap", "inverse_gap",
        "phys_wasserstein", "energy_violation",
        "contact_violation", "volume_violation",
        ("test_iron",  "success"),
        ("test_foam",  "success"),
        ("test_heavy", "success"),
        ("test_light", "success"),
    },

    # ── Legacy / backward-compat overrides (only used if you re-include
    # these baselines manually in --baselines).  Kept for reproducibility. ──
    "flat_vqvae":   {
        "closure_gap", "inverse_gap",
        ("test_iron",  "success"), ("test_foam",  "success"),
        ("test_heavy", "success"), ("test_light", "success"),
        "phys_wasserstein", "energy_violation",
        "contact_violation", "volume_violation",
    },
    "tamp_rule": {
        "closure_gap", "inverse_gap",
        "action_diversity", "result_diversity",
        ("test_iron", "success"), ("test_foam", "success"),
    },
    "_4dgs":     {
        "ade", "fde", "mpjpe", "closure_gap", "inverse_gap",
        "phys_wasserstein", "energy_violation",
        "contact_violation", "volume_violation",
        "success", "action_diversity", "result_diversity",
    },
}


def _is_na(baseline: str, split: str, field: str) -> bool:
    """Whether (baseline, split, field) cell should render as N/A.

    Supports both bare-field overrides (``"closure_gap"``) and split-specific
    overrides (``("test_high_friction", "success")``).
    """
    s = _NA_OVERRIDES.get(baseline, set())
    return (field in s) or ((split, field) in s)


# ══════════════════════════════════════════════════════════════════════
# §1  Wide summary-table layout (paper Table 1)
# ══════════════════════════════════════════════════════════════════════

# Each entry: (group_name, [(split_name, [(metric_field, header_label, fmt, lower_better), ...])])
# `split_name` is the JSON split key; `None` means a derived/computed metric.

# Table 1 (Reliability + Transfer) — algebraic gaps + trajectory error +
# success across all OOD slices.  This is the paper's main quantitative table.
_TABLE1_GROUPS: List[Tuple[str, List[Tuple[str, List[Tuple[str, str, str, bool]]]]]] = [
    ("IID", [
        ("test_iid", [
            ("closure_gap",  r"Clos $\downarrow$",    ".3f", True),
            ("inverse_gap",  r"Inv $\downarrow$",     ".3f", True),
            ("ade",          r"ADE $\downarrow$",     ".3f", True),
            ("fde",          r"FDE $\downarrow$",     ".3f", True),
            ("success",      r"Succ $\uparrow$",      ".2f", False),
        ]),
    ]),
    ("Unseen Pair", [
        ("test_ood_unseen_pair", [
            ("ade",     r"ADE $\downarrow$", ".3f", True),
            ("success", r"Succ $\uparrow$",  ".2f", False),
        ]),
    ]),
    ("Unseen Object", [
        ("test_ood_unseen_object", [
            ("ade",     r"ADE $\downarrow$", ".3f", True),
            ("success", r"Succ $\uparrow$",  ".2f", False),
        ]),
    ]),
    ("Unseen Act", [
        ("test_ood_unseen_action", [
            ("success", r"Succ $\uparrow$", ".2f", False),
        ]),
    ]),
    ("Cross-Material", [
        ("test_iron",  [("success", r"Iron $\uparrow$", ".2f", False)]),
        ("test_foam",  [("success", r"Foam $\uparrow$", ".2f", False)]),
        ("test_heavy", [("success", r"Heavy $\uparrow$", ".2f", False)]),
        ("test_light", [("success", r"Light $\uparrow$", ".2f", False)]),
    ]),
    ("Long horizon", [
        ("test_compositional_long", [
            ("success", r"Succ $\uparrow$", ".2f", False),
        ]),
    ]),
]


# Table 2 (Diversity + Physics Consistency) — quality dimensions per PDF
# metrics #9, #10, #11.  Reported per dataset.
_TABLE2_GROUPS: List[Tuple[str, List[Tuple[str, List[Tuple[str, str, str, bool]]]]]] = [
    ("Diversity (A)", [
        ("test_iid", [
            ("action_diversity", r"Lev $\uparrow$", ".2f", False),
            ("result_diversity", r"StateW $\uparrow$", ".2f", False),
        ]),
    ]),
    ("Diversity (B)", [
        ("test_iid", [          # for dataset_b, this is just "test"; iter handles it
            ("action_diversity", r"Lev $\uparrow$", ".2f", False),
            ("result_diversity", r"StateW $\uparrow$", ".2f", False),
        ]),
    ]),
    ("Physics Consistency", [
        ("test_iid", [
            ("phys_wasserstein",  r"Traj-W $\downarrow$",   ".3f", True),
            ("energy_violation",  r"Energy $\downarrow$",   ".2f", True),
            ("contact_violation", r"Contact $\downarrow$",  ".2f", True),
            ("volume_violation",  r"Volume $\downarrow$",   ".2f", True),
        ]),
    ]),
]


# Backward-compat alias (older code uses _SUMMARY_GROUPS)
_SUMMARY_GROUPS = _TABLE1_GROUPS


def _stat_value(stats: Dict, fmt: str) -> Optional[float]:
    if not stats or stats.get("n", 0) == 0:
        return None
    m = stats.get("mean")
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return None
    return float(m)


def _fmt_cell(stats: Dict, fmt: str, *, bold: bool = False, with_std: bool = False) -> str:
    if not stats or stats.get("n", 0) == 0:
        return "—"
    m = stats.get("mean"); s = stats.get("std", 0.0)
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "—"
    if with_std and s is not None and not (isinstance(s, float) and math.isnan(s)):
        cell = f"${m:{fmt}} \\!\\pm\\! {s:{fmt}}$"
    else:
        cell = f"${m:{fmt}}$"
    if bold:
        cell = f"\\textbf{{{cell}}}"
    return cell


def _find_best(values: List[Optional[float]], lower_is_better: bool) -> Optional[int]:
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    if lower_is_better:
        return min(valid, key=lambda iv: iv[1])[0]
    return max(valid, key=lambda iv: iv[1])[0]


def _build_table_from_groups(
    table_json: Dict,
    dataset:    str,
    groups:     List[Tuple[str, List[Tuple[str, List[Tuple[str, str, str, bool]]]]]],
    row_order:  List[str],
    with_std:   bool,
    table_label: str,
) -> str:
    """Build one wide table from a generic group spec.  Used by both
    build_table1 (reliability + transfer) and build_table2 (diversity + physics)."""
    # 1) Flatten all column descriptors
    columns: List[Tuple[str, str, str, str, bool]] = []   # (split, field, header, fmt, lower)
    group_widths: List[Tuple[str, int]] = []              # (group_name, n_cols)
    for group_name, splits in groups:
        n_in_group = 0
        for split, fields in splits:
            for field, header, fmt, lower in fields:
                columns.append((split, field, header, fmt, lower))
                n_in_group += 1
        group_widths.append((group_name, n_in_group))

    # 2) For each (column, baseline) → mean value (or None if N/A)
    cell_values: Dict[Tuple[int, str], Optional[float]] = {}
    for c_idx, (split, field, _, _, _) in enumerate(columns):
        for baseline in row_order:
            if _is_na(baseline, split, field):
                cell_values[(c_idx, baseline)] = None
                continue
            key = f"{baseline}::{dataset}::{split}"
            stats = table_json.get(key, {}).get(field, {})
            cell_values[(c_idx, baseline)] = _stat_value(stats, "")

    # 3) Bold-best per column
    bold_best: Dict[Tuple[int, str], bool] = {}
    for c_idx, (_, _, _, _, lower) in enumerate(columns):
        col_means = [cell_values[(c_idx, baseline)] for baseline in row_order]
        best = _find_best(col_means, lower)
        for r, baseline in enumerate(row_order):
            bold_best[(c_idx, baseline)] = (best == r)

    # 4) Render LaTeX
    n_cols = len(columns)
    align  = "l" + "c" * n_cols
    lines = [
        f"% Auto-generated {table_label} — dataset={dataset}",
        f"% (rows = baselines, columns = test conditions; bold = best per column)",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
    ]

    # Group header (multicolumn)
    header_parts = ["Method"]
    for g_name, n in group_widths:
        header_parts.append(f"\\multicolumn{{{n}}}{{c}}{{{g_name}}}")
    lines.append(" & ".join(header_parts) + r" \\")

    # cmidrule under each multicolumn group
    cmidrules = []
    col_cursor = 2     # column 1 is Method, groups start at column 2
    for _, n in group_widths:
        cmidrules.append(f"\\cmidrule(lr){{{col_cursor}-{col_cursor + n - 1}}}")
        col_cursor += n
    lines.append(" ".join(cmidrules))

    # Per-column metric labels
    metric_labels = [""] + [h for _, _, h, _, _ in columns]
    lines.append(" & ".join(metric_labels) + r" \\")
    lines.append("\\midrule")

    # Body
    for r, baseline in enumerate(row_order):
        if baseline == "ours":
            lines.append("\\midrule")
        cells = [_PAPER_NAME[baseline]]
        for c_idx, (split, field, _, fmt, _) in enumerate(columns):
            if _is_na(baseline, split, field):
                cells.append("—")
            else:
                key = f"{baseline}::{dataset}::{split}"
                stats = table_json.get(key, {}).get(field, {})
                bold = bold_best[(c_idx, baseline)]
                cells.append(_fmt_cell(stats, fmt, bold=bold, with_std=with_std))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


def build_table1(
    table_json: Dict,
    dataset:    str,
    row_order:  List[str] = _ROW_ORDER,
    with_std:   bool = False,
) -> str:
    """Paper Table 1: Reliability + Transfer.

    Columns: IID (Clos+Inv+ADE+FDE+Succ), Unseen-Pair (ADE+Succ),
    Unseen-Object (ADE+Succ), Unseen-Act (Succ), Cross-Material (4×Succ),
    Long-horizon (Succ).
    """
    return _build_table_from_groups(
        table_json, dataset, _TABLE1_GROUPS, row_order, with_std,
        table_label="paper Table 1 (Reliability + Transfer)",
    )


def build_table2(
    table_json: Dict,
    dataset:    str,
    row_order:  List[str] = _ROW_ORDER,
    with_std:   bool = False,
) -> str:
    """Paper Table 2: Diversity + Physics Consistency.

    Columns: Diversity (Lev + StateW per dataset),
    Physics (Wasserstein + Energy + Contact + Volume).
    """
    return _build_table_from_groups(
        table_json, dataset, _TABLE2_GROUPS, row_order, with_std,
        table_label="paper Table 2 (Diversity + Physics Consistency)",
    )


# Backward-compat alias — older code calls build_summary_table()
def build_summary_table(
    table_json: Dict,
    dataset:    str,
    row_order:  List[str] = _ROW_ORDER,
    with_std:   bool = False,
) -> str:
    return build_table1(table_json, dataset, row_order, with_std)


# ══════════════════════════════════════════════════════════════════════
# §2  Per-(dataset, split) detailed table (legacy / supplementary)
# ══════════════════════════════════════════════════════════════════════

_DETAILED_COLUMNS = [
    ("ade",              r"ADE $\downarrow$",     ".4f", True),
    ("fde",              r"FDE $\downarrow$",     ".4f", True),
    ("mpjpe",            r"MPJPE $\downarrow$",   ".4f", True),
    ("psnr",             r"PSNR $\uparrow$",      ".2f", False),
    ("lpips",            r"LPIPS $\downarrow$",   ".3f", True),
    ("closure_gap",      r"Clos $\downarrow$",    ".4f", True),
    ("inverse_gap",      r"Inv $\downarrow$",     ".4f", True),
    ("energy_violation", r"Energy $\downarrow$",  ".3f", True),
    ("success",          r"Succ $\uparrow$",      ".3f", False),
]


def build_detailed_table(
    table_json: Dict,
    dataset:    str,
    split:      str,
    row_order:  List[str] = _ROW_ORDER,
    with_std:   bool = True,
) -> str:
    """One LaTeX table per (dataset, split) — useful for the appendix."""
    cell_values: Dict[Tuple[int, str], Optional[float]] = {}
    for c_idx, (field, _, _, _) in enumerate(_DETAILED_COLUMNS):
        for baseline in row_order:
            if _is_na(baseline, split, field):
                cell_values[(c_idx, baseline)] = None
                continue
            key = f"{baseline}::{dataset}::{split}"
            stats = table_json.get(key, {}).get(field, {})
            cell_values[(c_idx, baseline)] = _stat_value(stats, "")

    bold_best: Dict[Tuple[int, str], bool] = {}
    for c_idx, (_, _, _, lower) in enumerate(_DETAILED_COLUMNS):
        col_means = [cell_values[(c_idx, baseline)] for baseline in row_order]
        best = _find_best(col_means, lower)
        for r, baseline in enumerate(row_order):
            bold_best[(c_idx, baseline)] = (best == r)

    n_cols = len(_DETAILED_COLUMNS)
    align  = "l" + "c" * n_cols
    lines = [
        f"% Per-split detailed table — dataset={dataset}, split={split}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        "Method & " + " & ".join(h for _, h, _, _ in _DETAILED_COLUMNS) + r" \\",
        "\\midrule",
    ]
    for r, baseline in enumerate(row_order):
        if baseline == "ours":
            lines.append("\\midrule")
        cells = [_PAPER_NAME[baseline]]
        for c_idx, (field, _, fmt, _) in enumerate(_DETAILED_COLUMNS):
            if _is_na(baseline, split, field):
                cells.append("—")
            else:
                key = f"{baseline}::{dataset}::{split}"
                stats = table_json.get(key, {}).get(field, {})
                cells.append(_fmt_cell(stats, fmt, bold=bold_best[(c_idx, baseline)],
                                         with_std=with_std))
        lines.append(" & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", required=True,
                   help="path to main_table.json from aggregate.py")
    p.add_argument("--dataset", required=True,
                   help="which dataset (dataset_a / dataset_b)")

    # Three output modes
    p.add_argument("--two-tables", action="store_true",
                   help="(recommended) output Table 1 (reliability+transfer) "
                        "and Table 2 (diversity+physics) — paper main tables. "
                        "Use --output1 and --output2 to split into two files.")
    p.add_argument("--summary", action="store_true",
                   help="single wide table (alias of --two-tables --table 1; legacy)")
    p.add_argument("--table", choices=["1", "2"], default=None,
                   help="with --two-tables: which single table to output (default: both)")

    p.add_argument("--with-std", action="store_true",
                   help="include ±std in cells (default: mean only for compactness)")
    p.add_argument("--splits", nargs="+", default=None,
                   help="for per-split (default) mode: which splits to render")
    p.add_argument("--output",  default=None,
                   help=".tex output path (default: stdout). For --two-tables "
                        "when both tables are output, both go in this single file "
                        "separated by a blank line.")
    p.add_argument("--output1", default=None,
                   help="separate file for Table 1 (used with --two-tables)")
    p.add_argument("--output2", default=None,
                   help="separate file for Table 2 (used with --two-tables)")
    args = p.parse_args(argv)

    with open(args.json) as f:
        table_json = json.load(f)

    if args.two_tables or args.summary:
        which = args.table  # "1", "2", or None=both
        chunks = []
        if which in (None, "1"):
            t1 = build_table1(table_json, args.dataset, with_std=args.with_std)
            chunks.append(("table1", t1))
        if which in (None, "2"):
            t2 = build_table2(table_json, args.dataset, with_std=args.with_std)
            chunks.append(("table2", t2))

        # Per-table output files, if specified
        if args.output1 and any(name == "table1" for name, _ in chunks):
            Path(args.output1).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output1).write_text(dict(chunks)["table1"])
            print(f"✓ wrote Table 1 → {args.output1}")
        if args.output2 and any(name == "table2" for name, _ in chunks):
            Path(args.output2).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output2).write_text(dict(chunks)["table2"])
            print(f"✓ wrote Table 2 → {args.output2}")

        # Combined output (or stdout)
        if args.output1 is None and args.output2 is None:
            out = "\n\n".join(c for _, c in chunks)
        else:
            out = ""  # already wrote individual files
    else:
        if args.splits is None:
            splits = sorted({
                k.split("::")[2] for k in table_json
                if not k.startswith("_") and f"::{args.dataset}::" in k
            })
        else:
            splits = args.splits
        if not splits:
            print(f"✗ no entries for dataset={args.dataset!r}", file=sys.stderr)
            return 1
        chunks = []
        for split in splits:
            chunks.append(build_detailed_table(table_json, args.dataset, split,
                                                with_std=args.with_std))
            chunks.append("")
        out = "\n".join(chunks)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(out)
        print(f"✓ wrote {args.output}")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
