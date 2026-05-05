"""Step 5: Build train/val/test splits and write manifest for Dataset-B.

Reads:  outputs/data/* (their meta.json), configs/default.yaml
Writes: outputs/splits.json, outputs/manifest.json, outputs/trajectories.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.packager import write_manifest
from src.splitter import assert_no_leakage, save_splits, split_by_verb


def _load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      default="outputs/data")
    ap.add_argument("--config",        default="configs/default.yaml")
    ap.add_argument("--out_splits",    default="outputs/splits.json")
    ap.add_argument("--out_manifest",  default="outputs/manifest.json")
    ap.add_argument("--out_trajectories", default="outputs/trajectories.json")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = _load_yaml(Path(args.config))
    splits_cfg = cfg["splits"]

    # Read each completed trajectory's meta.json
    data_dir = Path(args.data_dir)
    records = []
    for tdir in sorted(data_dir.iterdir()):
        if not tdir.is_dir():
            continue
        mp = tdir / "meta.json"
        if not mp.exists():
            continue
        # Need init_gs.ply to consider it a complete record (Step 4 done)
        if not (tdir / "init_gs.ply").exists():
            logging.warning("skipping %s: no init_gs.ply (Step 4 incomplete?)", tdir.name)
            continue
        with open(mp) as f:
            records.append(json.load(f))
    logging.info("Found %d complete trajectories in %s", len(records), data_dir)

    # Save trajectories.json (full record list, keeps everything from meta.json)
    out_traj = Path(args.out_trajectories)
    out_traj.parent.mkdir(parents=True, exist_ok=True)
    with open(out_traj, "w") as f:
        json.dump({"trajectories": records, "n": len(records)}, f, indent=2)
    logging.info("Wrote %s", out_traj)

    # Stratified split per verb
    splits = split_by_verb(
        records,
        train_frac=splits_cfg["train_frac"],
        val_frac=splits_cfg["val_frac"],
        test_frac=splits_cfg["test_frac"],
        seed=splits_cfg.get("seed", 42),
    )
    assert_no_leakage(splits)
    save_splits(splits, args.out_splits, meta={"config": args.config})

    write_manifest(args.data_dir, splits, args.out_manifest)
    print("Done.")


if __name__ == "__main__":
    main()
