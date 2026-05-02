#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def check_internal_imports() -> None:
    modules = [
        "assets.loader",
        "sim.rigid_pybullet",
        "sim.deform_taichi",
        "render.blender_runner",
        "tokens.extract",
        "pairs.sample_pairs",
        "splits.build_splits",
        "pipeline.build_episode",
    ]
    for module in modules:
        importlib.import_module(module)


def check_configs() -> None:
    required = [
        "object_catalog.yaml",
        "action_vocab.yaml",
        "physics_profiles.yaml",
        "render.yaml",
    ]
    missing = [name for name in required if not (ROOT / "configs" / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing configs: {missing}")


def check_determinism() -> None:
    from common.random_state import seed_from_episode_id

    first = seed_from_episode_id("ep_000001", 123)
    second = seed_from_episode_id("ep_000001", 123)
    if first != second:
        raise AssertionError("Episode seed factory is not deterministic")


def main() -> None:
    check_internal_imports()
    check_configs()
    check_determinism()
    print("OK")


if __name__ == "__main__":
    main()
