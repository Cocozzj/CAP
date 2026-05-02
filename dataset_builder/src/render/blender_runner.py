from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderJob:
    episode_dir: Path
    cameras_json: Path
    frames: int = 60
    views: int = 3


class BlenderRenderer:
    def __init__(self, resolution: tuple[int, int] = (512, 512)) -> None:
        self.resolution = resolution

    def render_episode(self, job: RenderJob) -> None:
        try:
            import bpy  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Blender Python API is required for rendering; run this inside Blender or install bpy"
            ) from exc
        _ = bpy
        raise NotImplementedError("Blender rendering is implemented in Phase 1")
