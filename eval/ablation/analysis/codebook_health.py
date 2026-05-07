"""codebook_health.py — Per-variant codebook utilisation diagnostics.

Loads each variant's main_exp_final.pt, extracts the action-codebook tensor,
and reports utilisation statistics:

  - K              : codebook size
  - unique_rows    : count of distinct codes (rounded to 3 decimals)
  - frac_used      : unique / K  (1.0 = healthy, 0.0x = collapse)
  - norm_mean/std  : per-row L2 norm distribution
  - entropy_bits   : effective entropy from codebook usage variance

A collapsed codebook (frac_used << 1) is the smoking gun for a failed
ablation — usually means the variant ablation kept too few signals to
keep all codes alive.

Output:
  <out_dir>/codebook_health.csv   ← raw stats
  <out_dir>/codebook_health.md    ← Tab S3 candidate

Usage::
    python eval/ablation/analysis/codebook_health.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch


def find_codebook(state_dict: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """Return the action-codebook weight tensor, or None if absent."""
    for k in state_dict:
        if "action_enc" in k and "codebook.weight" in k:
            return state_dict[k]
    return None


def codebook_stats(cb: torch.Tensor, round_dec: int = 3) -> Dict:
    """One-shot computation of all metrics we'll display."""
    K, dim = cb.shape
    norms = cb.float().norm(dim=-1)
    # Approx unique rows by rounding (cheaper than full set, robust to fp jitter).
    unique = torch.unique(cb.float().round(decimals=round_dec), dim=0).shape[0]
    return {
        "K":            int(K),
        "dim":          int(dim),
        "unique_rows":  int(unique),
        "frac_used":    unique / float(K),
        "norm_mean":    float(norms.mean()),
        "norm_std":     float(norms.std(unbiased=False)),
        "norm_min":     float(norms.min()),
        "norm_max":     float(norms.max()),
    }


def collect_ckpts(run_root: Path) -> Dict[str, Path]:
    out = {}
    if not run_root.exists():
        return out
    for variant_dir in sorted(run_root.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name.startswith("_"):
            continue
        ck = variant_dir / "seed_0" / "ckpt" / "main_exp_final.pt"
        if ck.exists():
            out[variant_dir.name] = ck
    return out


def collect_main_ckpts(seeds=(0, 1, 2)) -> Dict[str, Path]:
    out = {}
    for s in seeds:
        for sub in (Path("runs/main_a"), Path("runs/main_exp")):
            ck = sub / f"seed_{s}" / "ckpt" / "main_exp_final.pt"
            if ck.exists():
                out[f"main_seed{s}"] = ck
                break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--module-root", type=str, default="runs/module")
    p.add_argument("--loss-root",   type=str, default="runs/loss")
    p.add_argument("--main-seeds",  nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--out-dir",     type=str, default="runs/_analysis")
    p.add_argument("--device",      type=str, default="cpu",
                   help="cpu is fine — only loading codebook (small tensor)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: Dict[str, Path] = {}
    sources.update(collect_main_ckpts(args.main_seeds))
    sources.update(collect_ckpts(Path(args.module_root)))
    sources.update(collect_ckpts(Path(args.loss_root)))

    print(f"=== Loading {len(sources)} ckpts (codebook only) ===")
    stats: Dict[str, Dict] = {}
    for name, ck in sorted(sources.items()):
        try:
            state = torch.load(ck, map_location=args.device, weights_only=False)
        except Exception as e:
            print(f"  ✗ {name}: load failed — {e}")
            continue
        sd = state.get("model", state)
        cb = find_codebook(sd)
        if cb is None:
            print(f"  ✗ {name}: no codebook tensor in state_dict")
            continue
        st = codebook_stats(cb)
        stats[name] = st
        marker = "✓" if st["frac_used"] > 0.5 else ("⚠" if st["frac_used"] > 0.1 else "✗")
        print(f"  {marker} {name:20s}  K={st['K']:>5d}  "
              f"unique={st['unique_rows']:>5d}/{st['K']:<5d}  "
              f"({100*st['frac_used']:>5.1f}%)  "
              f"norm={st['norm_mean']:.3f}±{st['norm_std']:.3f}")

    # ── CSV ──
    csv_path = out_dir / "codebook_health.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "K", "unique_rows", "frac_used",
                    "norm_mean", "norm_std", "norm_min", "norm_max"])
        for name, st in sorted(stats.items()):
            w.writerow([name, st["K"], st["unique_rows"],
                        f"{st['frac_used']:.4f}",
                        f"{st['norm_mean']:.4f}",
                        f"{st['norm_std']:.4f}",
                        f"{st['norm_min']:.4f}",
                        f"{st['norm_max']:.4f}"])
    print(f"\n  ✔ wrote {csv_path}")

    # ── Markdown ──
    md_path = out_dir / "codebook_health.md"
    md = ["# Codebook utilisation across ablation variants\n",
          "Generated by `eval/ablation/analysis/codebook_health.py`. ",
          "`frac_used` is the fraction of codebook rows that are distinct (≈ "
          "fraction of codes that are not collapsed to a common value). "
          "Healthy ≈ 1.00; full collapse → near 0.\n"]

    md.append("| variant | K | unique | frac used | norm mean ± std | min | max | health |")
    md.append("|---|---|---|---|---|---|---|---|")
    main_keys = [k for k in stats if k.startswith("main_seed")]
    other_keys = [k for k in stats if not k.startswith("main_seed")]
    for name in sorted(main_keys) + sorted(other_keys):
        st = stats[name]
        if st["frac_used"] > 0.95:
            health = "✅ healthy"
        elif st["frac_used"] > 0.50:
            health = "🟡 partial"
        elif st["frac_used"] > 0.10:
            health = "🟠 degraded"
        else:
            health = "🔴 collapsed"
        md.append(
            f"| {name} | {st['K']} | {st['unique_rows']} | "
            f"{100*st['frac_used']:.1f}% | "
            f"{st['norm_mean']:.3f} ± {st['norm_std']:.3f} | "
            f"{st['norm_min']:.3f} | {st['norm_max']:.3f} | {health} |"
        )

    md.append("\n## How to read this")
    md.append("- `frac_used = 100%` → all codes carry distinct content (ideal).")
    md.append("- `frac_used < 50%` → many codes settled to identical values "
              "(VQ commitment loss couldn't differentiate them).")
    md.append("- `norm_std` near 0 with `frac_used = 100%` → codes diverse "
              "but uniform-magnitude (likely lattice init + light fine-tune).")
    md.append("- `norm_std` large with `frac_used < 50%` → some active codes "
              "(large norm) coexist with dead codes (small norm) — bimodal "
              "collapse.")

    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"  ✔ wrote {md_path}")


if __name__ == "__main__":
    main()
