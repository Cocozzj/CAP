from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_phase0_episode_stub_schema(tmp_path: Path) -> None:
    from pipeline.build_episode import EpisodeRequest, build_episode_stub

    episode_dir = build_episode_stub(
        EpisodeRequest(
            episode_id="ep_000001",
            object_id="drawer_03",
            action="open",
            output_root=tmp_path,
            base_seed=123,
        )
    )

    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["episode_id"] == "ep_000001"
    assert meta["duration_frames"] == 60
    assert meta["num_views"] == 3
    assert meta["atomic_seq_len"] == 12

    tokens = np.load(episode_dir / "action_tokens.npz")
    assert tokens["ell"].shape == (12, 3)
    assert tokens["h"].shape == (12, 4)
    assert tokens["xi"].shape == (12, 6)
    assert tokens["rho"].shape == (12, 8)
    assert tokens["token_id"].dtype == np.int32
