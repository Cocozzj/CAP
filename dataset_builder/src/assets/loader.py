from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.config import expand_asset_root, load_config
from common.paths import resolve_repo_path


@dataclass(frozen=True)
class AssetSpec:
    object_id: str
    object_class: str
    asset: str
    asset_path: Path
    kind: str
    affordances: tuple[str, ...]
    default_physics_profile: str


class AssetCatalog:
    def __init__(self, catalog: dict[str, Any] | None = None) -> None:
        raw = expand_asset_root(catalog or load_config("object_catalog.yaml"))
        self.asset_root = resolve_repo_path(raw["asset_root"])
        self._objects = {
            item["id"]: AssetSpec(
                object_id=item["id"],
                object_class=item["class"],
                asset=item["asset"],
                asset_path=resolve_repo_path(item["asset_path"]),
                kind=item["kind"],
                affordances=tuple(item["affordances"]),
                default_physics_profile=item["default_physics_profile"],
            )
            for item in raw["objects"]
        }

    def get(self, object_id: str) -> AssetSpec:
        return self._objects[object_id]

    def all(self) -> list[AssetSpec]:
        return list(self._objects.values())


def load_partnet_urdf(asset: AssetSpec) -> Path:
    candidates = sorted(asset.asset_path.glob("*.urdf"))
    if not candidates:
        raise FileNotFoundError(f"No URDF found for {asset.object_id}: {asset.asset_path}")
    return candidates[0]
