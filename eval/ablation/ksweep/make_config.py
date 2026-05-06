"""make_config.py — Patch a base config.yaml for a single K value.

Loads a base CAP config, overrides ``encoder.action_tokenizer.num_action_codebook``
to the requested K, and writes the result to a new YAML file.

Usage::

    python eval/ablation/ksweep/make_config.py \\
        --base configs/config.yaml --K 256 \\
        --out  configs/_ksweep/config_K256.yaml

The base config has YAML anchors (``&motion_dim``, ``&atomic_dim`` etc.).
``yaml.safe_load`` resolves them to literal values, and ``yaml.safe_dump``
writes them inlined — that's still valid YAML and trainer reads it identically.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def patch_K(base_cfg_path: Path, K: int, out_path: Path) -> None:
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Overwrite the codebook size — single source of truth, all downstream
    # derivations (k_prim, EOS slot id, etc.) read from this in model.py:66.
    enc = cfg.get("encoder", {})
    at = enc.get("action_tokenizer")
    if at is None:
        raise SystemExit(
            f"{base_cfg_path}: encoder.action_tokenizer not found — "
            "config schema may have changed; update make_config.py."
        )
    old_K = at.get("num_action_codebook")
    at["num_action_codebook"] = int(K)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        # sort_keys=False preserves the schema's logical order (gs_param →
        # encoder → planner → executor) for readability.  default_flow_style
        # writes block style not the inline {a: 1, b: 2} form.
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    print(f"[make_config] base={base_cfg_path}  K: {old_K} → {K}  out={out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=str, default="configs/config.yaml",
                   help="Base config to clone (default: configs/config.yaml)")
    p.add_argument("--K",    type=int, required=True,
                   help="Atomic codebook size to inject")
    p.add_argument("--out",  type=str, required=True,
                   help="Output YAML path (parent dirs will be created)")
    args = p.parse_args()
    patch_K(Path(args.base), args.K, Path(args.out))


if __name__ == "__main__":
    main()
