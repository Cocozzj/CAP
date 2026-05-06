"""make_config.py — generate per-variant config + loss-config + flags.

Reads ``variants.py`` and emits, for the requested variant:

  <out_dir>/config.yaml         — patched main config
  <out_dir>/loss.yaml           — patched A-side loss config
  <out_dir>/loss_b.yaml         — patched B-side loss config
  <out_dir>/trainer_flags.txt   — extra CLI flags (one per line)
  <out_dir>/description.txt     — human-readable description

Usage::

    python eval/ablation/module/make_config.py --variant no_algebraic \\
        --base-config configs/config.yaml \\
        --base-loss-a configs/loss.yaml   --base-loss-b configs/loss_b.yaml \\
        --out-dir configs/_ablation/no_algebraic

Path overrides use dot-notation in variants.py:
  ``"loss.lambda_clos": 0.0``   →   cfg["loss"]["lambda_clos"] = 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Make ``variants`` importable when running this script as a standalone file
# (from CAP/ root we'd `python eval/ablation/module/make_config.py`, so the
# directory of this file is what we prepend).
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import variants as variants_mod   # noqa: E402


def _set_dotpath(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """In-place set for a nested key path like ``a.b.c``.

    Creates intermediate dicts if missing — but warns on unexpected creation
    since the schema should already define the key.
    """
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _dump_yaml(d: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False, default_flow_style=False)


def make_variant(
    variant_name: str,
    base_config:  Path,
    base_loss_a:  Path,
    base_loss_b:  Path,
    out_dir:      Path,
) -> Dict[str, Any]:
    """Materialise one variant's configs.  Returns the variant spec for caller use."""
    spec = variants_mod.get_variant(variant_name)
    cfg_overrides:  Dict[str, Any] = spec.get("config_overrides", {})
    loss_overrides: Dict[str, Any] = spec.get("loss_overrides",   {})
    flags:          List[str]      = spec.get("trainer_flags",    [])
    desc:           str            = spec.get("description",      variant_name)

    # ── main config ────────────────────────────────────────────────
    cfg = _load_yaml(base_config)
    for k, v in cfg_overrides.items():
        _set_dotpath(cfg, k, v)
    _dump_yaml(cfg, out_dir / "config.yaml")

    # ── A-side loss config ─────────────────────────────────────────
    loss_a = _load_yaml(base_loss_a)
    for k, v in loss_overrides.items():
        _set_dotpath(loss_a, k, v)
    _dump_yaml(loss_a, out_dir / "loss.yaml")

    # ── B-side loss config (same overrides — semantics carry over) ─
    loss_b = _load_yaml(base_loss_b)
    for k, v in loss_overrides.items():
        _set_dotpath(loss_b, k, v)
    _dump_yaml(loss_b, out_dir / "loss_b.yaml")

    # ── trainer flag list ──────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "trainer_flags.txt", "w") as f:
        for fl in flags:
            f.write(f"{fl}\n")

    # ── human-readable description for log review ──────────────────
    with open(out_dir / "description.txt", "w") as f:
        f.write(f"variant: {variant_name}\n")
        f.write(f"description:\n  {desc}\n\n")
        f.write("config_overrides:\n")
        for k, v in cfg_overrides.items():  f.write(f"  {k}: {v}\n")
        f.write("loss_overrides:\n")
        for k, v in loss_overrides.items(): f.write(f"  {k}: {v}\n")
        f.write("trainer_flags:\n")
        for fl in flags:                    f.write(f"  {fl}\n")

    print(f"[make_config] {variant_name}  → {out_dir}")
    print(f"  config_overrides: {len(cfg_overrides)} key(s)")
    print(f"  loss_overrides:   {len(loss_overrides)} key(s)")
    print(f"  trainer_flags:    {flags or '(none)'}")
    return spec


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant",     type=str, required=True,
                   help=f"Name from variants.py — one of: {variants_mod.list_variants()}")
    p.add_argument("--base-config", type=str, default="configs/config.yaml")
    p.add_argument("--base-loss-a", type=str, default="configs/loss.yaml")
    p.add_argument("--base-loss-b", type=str, default="configs/loss_b.yaml")
    p.add_argument("--out-dir",     type=str, required=True)
    args = p.parse_args()

    make_variant(
        variant_name = args.variant,
        base_config  = Path(args.base_config),
        base_loss_a  = Path(args.base_loss_a),
        base_loss_b  = Path(args.base_loss_b),
        out_dir      = Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
