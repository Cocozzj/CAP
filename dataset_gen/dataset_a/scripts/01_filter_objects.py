"""Step 1: Enumerate PartNet-Mobility, filter by category whitelist + motion saliency.

Output: object_list.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.object_selector import (
    load_yaml,
    save_object_list,
    select_objects,
)


def main():
    ap = argparse.ArgumentParser(description="Filter PartNet-Mobility for Dataset-A")
    ap.add_argument("--partnet_root", required=True,
                    help="Path to PartNet-Mobility extraction (folder containing dataset/<obj_id>/...)")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="Main config")
    ap.add_argument("--categories", default="configs/object_categories.yaml",
                    help="Category whitelist + per-category rules")
    ap.add_argument("--out", default="outputs/object_list.json",
                    help="Where to save the filtered object list")
    ap.add_argument("--no_saliency", action="store_true",
                    help="Skip the motion-saliency render step (faster, less accurate)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_yaml(args.config)
    cat_cfg = load_yaml(args.categories)

    saliency_cfg = dict(cfg["saliency"])
    if args.no_saliency:
        saliency_cfg["enable"] = False

    selected = select_objects(
        partnet_root=args.partnet_root,
        category_cfg=cat_cfg,
        saliency_cfg=saliency_cfg,
        instances_per_category=cfg["scale"]["instances_per_category"],
        verbose=True,
    )
    save_object_list(selected, args.out)
    print(f"Done. {len(selected)} objects → {args.out}")


if __name__ == "__main__":
    main()
