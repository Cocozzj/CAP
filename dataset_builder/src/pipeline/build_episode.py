from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from assets.loader import AssetCatalog
from common.random_state import random_state_for_episode, seed_from_episode_id
from tokens.extract import empty_action_tokens, save_action_tokens


@dataclass(frozen=True)
class EpisodeRequest:
    episode_id: str
    object_id: str
    action: str
    output_root: Path
    split: str = "train"
    base_seed: int = 0


def build_episode_stub(request: EpisodeRequest) -> Path:
    """Create metadata and token placeholders used by Phase 0 smoke checks."""
    catalog = AssetCatalog()
    asset = catalog.get(request.object_id)
    episode_dir = request.output_root / "datasetA_synth" / "episodes" / request.episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)

    seed = seed_from_episode_id(request.episode_id, request.base_seed)
    rng = random_state_for_episode(request.episode_id, request.base_seed)
    c_task_id = int(rng.randint(0, 128))
    meta = {
        "episode_id": request.episode_id,
        "dataset": "A_synth",
        "object": {"id": asset.object_id, "class": asset.object_class, "asset": asset.asset},
        "task": {"label": f"{request.action}_{asset.object_class}", "c_task_id": c_task_id},
        "duration_frames": 60,
        "fps": 60,
        "num_views": 3,
        "atomic_seq_len": 12,
        "pair_role": "primary",
        "pair_partners": {},
        "physics_profile": asset.default_physics_profile,
        "split": request.split,
        "seed": seed,
    }
    with (episode_dir / "meta.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")
    save_action_tokens(str(episode_dir / "action_tokens.npz"), empty_action_tokens(c_task_id=c_task_id))
    return episode_dir
