from __future__ import annotations

import hashlib

import numpy as np


def seed_from_episode_id(episode_id: str, base_seed: int = 0) -> int:
    digest = hashlib.blake2b(
        f"{base_seed}:{episode_id}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def random_state_for_episode(episode_id: str, base_seed: int = 0) -> np.random.RandomState:
    return np.random.RandomState(seed_from_episode_id(episode_id, base_seed))
