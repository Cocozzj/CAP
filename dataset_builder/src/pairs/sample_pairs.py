from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PAIR_FILES = {
    "closure": "closure_pairs.json",
    "inverse": "inverse_pairs.json",
    "commutator": "commutator_pairs.json",
    "transfer": "transfer_pairs.json",
}


def write_pair_file(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")


def validate_pair_refs(entries: list[dict[str, Any]], episode_ids: set[str]) -> list[str]:
    missing: list[str] = []
    for entry in entries:
        for key, value in entry.items():
            if key.startswith("ep_") and isinstance(value, str) and value not in episode_ids:
                missing.append(value)
    return missing
