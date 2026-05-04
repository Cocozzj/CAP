"""Step 5: Build train/val/test splits and write manifest (+ optional WebDataset shards).

Reads:  trajectories.json + default.yaml + outputs/data/
Writes: splits.json + manifest.json (+ optional shards/)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.object_selector import load_yaml
from src.packager import pack_webdataset, write_manifest
from src.splitter import (
    assert_no_leakage,
    save_splits,
    split_trajectories,
)
from src.trajectory_generator import load_trajectories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--data_dir", required=True,
                    help="Output dir from Step 4")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_splits", default="outputs/splits.json")
    ap.add_argument("--out_manifest", default="outputs/manifest.json")
    ap.add_argument("--out_shards", default=None,
                    help="If set, also pack WebDataset shards here")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_yaml(args.config)
    splits_cfg = cfg["splits"]
    pkg_cfg = cfg.get("packaging", {})

    records = load_trajectories(args.trajectories)
    splits = split_trajectories(
        records,
        train_frac=splits_cfg["train_frac"],
        val_frac=splits_cfg["val_frac"],
        test_iid_frac=splits_cfg["test_iid_frac"],
        ood_pair_fraction=splits_cfg["ood_pair_fraction"],
        held_out_categories=splits_cfg.get("held_out_categories", []),
        seed=cfg["runtime"]["seed"],
    )
    assert_no_leakage(splits, records)
    save_splits(splits, args.out_splits, meta={"config": args.config})

    write_manifest(args.data_dir, splits, args.out_manifest)

    if args.out_shards or pkg_cfg.get("format") == "webdataset":
        out_shards = args.out_shards or "outputs/shards"
        pack_webdataset(args.data_dir, splits, out_shards,
                        shards_per_split=pkg_cfg.get("shards_per_split", 40))

    print("Done.")


if __name__ == "__main__":
    main()
