#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipeline.build_episode import EpisodeRequest, build_episode_stub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one Dataset-A episode.")
    parser.add_argument("--episode-id", default="ep_000001")
    parser.add_argument("--object-id", default="drawer_03")
    parser.add_argument("--action", default="open")
    parser.add_argument("--output-root", default=str(ROOT.parent / "4DTokenizer-Dataset"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir = build_episode_stub(
        EpisodeRequest(
            episode_id=args.episode_id,
            object_id=args.object_id,
            action=args.action,
            output_root=Path(args.output_root),
            base_seed=args.seed,
        )
    )
    print(f"Wrote Phase 0 episode stub: {episode_dir}")


if __name__ == "__main__":
    main()
