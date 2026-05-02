from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from common.paths import CONFIG_DIR


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return data


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def expand_asset_root(catalog: dict[str, Any]) -> dict[str, Any]:
    asset_root = str(catalog.get("asset_root", ""))
    for obj in catalog.get("objects", []):
        asset_path = obj.get("asset_path")
        if isinstance(asset_path, str):
            obj["asset_path"] = asset_path.replace("${asset_root}", asset_root)
    return catalog
