"""Step 2: Generate the (object × task × seed) plan and run each trajectory in
SAPIEN to obtain joint qpos sequences.

Atomic trajectories are always produced. If --compositions points at a
compositions.yaml, 2-step compositions (in-train) and 3-step+ compositions
(eval-only) are added on top.

Reads:
  - object_list.json     (Step 1)
  - configs/tasks.yaml
  - configs/compositions.yaml  (optional)
  - configs/default.yaml

Writes:
  - trajectories.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.object_selector import load_yaml
from src.trajectory_generator import (
    generate_all_trajectories,
    save_trajectories,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object_list", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tasks", default="configs/tasks.yaml")
    ap.add_argument("--compositions", default="configs/compositions.yaml",
                    help="Compositions config; pass '' to disable compositions entirely")
    ap.add_argument("--out", default="outputs/trajectories.json")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Parallel CPU workers (each runs its own SAPIEN engine). "
                         "Defaults to 1 (serial). Try 8-16 on a beefy CPU.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_yaml(args.config)
    tasks_cfg = load_yaml(args.tasks)

    compositions_cfg = None
    if args.compositions:
        comp_path = Path(args.compositions)
        if comp_path.exists():
            compositions_cfg = load_yaml(comp_path)
            logging.info("Loaded compositions from %s", comp_path)
        else:
            logging.warning("Compositions config %s not found; running atomic-only", comp_path)

    with open(args.object_list) as f:
        object_list = json.load(f)["objects"]

    records = generate_all_trajectories(
        object_list=object_list,
        tasks_cfg=tasks_cfg,
        physics_cfg=cfg["physics"],
        trajectories_per_pair=cfg["scale"]["trajectories_per_pair"],
        target_total=cfg["scale"]["target_total_trajectories"],
        fps=cfg["render"]["fps"],
        seed=cfg["runtime"]["seed"],
        compositions_cfg=compositions_cfg,
        num_workers=args.num_workers,
    )
    save_trajectories(records, args.out)
    print(f"Done. {len(records)} trajectories → {args.out}")


if __name__ == "__main__":
    main()
