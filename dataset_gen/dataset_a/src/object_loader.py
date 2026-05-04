"""SAPIEN-based loader for PartNet-Mobility objects.

PartNet-Mobility folder layout (typical):

    <partnet_root>/
        dataset/
            <obj_id>/                 # numeric, e.g. 100147
                mobility.urdf
                mobility_v2.json      # joint metadata
                meta.json             # {"model_cat": "Door", ...}
                textured_objs/
                ...

Use `enumerate_partnet_objects()` to scan; use `PartNetObject.load()` to bring
one into a SAPIEN scene.
"""

from __future__ import annotations

import json
import math
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

# SAPIEN is heavy and depends on a working GL/Vulkan backend; we import lazily
# so that scripts that only need metadata (e.g. enumerate) can run on a CPU
# box without it.
def _lazy_sapien():
    import sapien.core as sapien  # noqa: WPS433  (intentional lazy import)
    return sapien


@dataclass
class JointInfo:
    name: str
    joint_type: str          # "revolute" | "prismatic" | "fixed" | "continuous"
    limit_low: float
    limit_high: float
    axis: List[float]
    parent_link: str
    child_link: str

    @property
    def range(self) -> float:
        return self.limit_high - self.limit_low

    @property
    def is_active(self) -> bool:
        return self.joint_type in ("revolute", "prismatic", "continuous")


@dataclass
class PartNetObject:
    """Lightweight metadata wrapper for one PartNet-Mobility instance.

    Real SAPIEN articulation objects are created on-demand via `.load()`.
    """

    obj_id: str
    folder: Path
    category: str                                  # from meta.json["model_cat"]
    joints: List[JointInfo] = field(default_factory=list)
    bbox_diagonal: float = 0.0
    has_textures: bool = False

    @classmethod
    def from_folder(cls, folder: Path, *, compute_bbox: bool = False) -> Optional["PartNetObject"]:
        """Read metadata from disk; returns None if the folder is malformed.

        `compute_bbox=False` (default) skips loading mesh files via trimesh
        and just sets bbox_diagonal=1.0 (cheap proxy). Bbox is only needed at
        render time, not at enumeration time.
        """
        try:
            meta_path = folder / "meta.json"
            mobility_path = folder / "mobility_v2.json"
            urdf_path = folder / "mobility.urdf"
            if not (meta_path.exists() and urdf_path.exists()):
                return None

            with open(meta_path) as f:
                meta = json.load(f)
            category = meta.get("model_cat") or meta.get("category")
            if category is None:
                return None

            joints = []
            if mobility_path.exists():
                with open(mobility_path) as f:
                    mobility = json.load(f)
                # PartNet-Mobility's actual schema:
                #   entry = {
                #     "id": int, "parent": int, "name": str,
                #     "joint": "hinge" | "slider" | "fixed" | ...,    <-- STRING
                #     "jointData": {
                #       "axis": {"origin": [...], "direction": [...]},
                #       "limit": {"a": float, "b": float, "noLimit": bool}
                #     },
                #     "parts": [...]
                #   }
                # Map PartNet joint type name -> URDF-style name we use elsewhere
                type_map = {
                    "hinge":      "revolute",
                    "slider":     "prismatic",
                    "revolute":   "revolute",
                    "prismatic":  "prismatic",
                    "continuous": "continuous",
                }
                for entry in mobility:
                    if not isinstance(entry, dict):
                        continue
                    raw_joint = entry.get("joint")
                    # Accept both legacy (dict) and current (string) shapes
                    if isinstance(raw_joint, dict):
                        jtype = raw_joint.get("type", "fixed")
                        joint_data = raw_joint
                    elif isinstance(raw_joint, str):
                        jtype = type_map.get(raw_joint, "fixed")
                        joint_data = entry.get("jointData", {})
                    else:
                        continue
                    if jtype not in ("revolute", "prismatic", "continuous"):
                        continue
                    if not isinstance(joint_data, dict):
                        continue
                    limit = joint_data.get("limit", {})
                    if not isinstance(limit, dict):
                        limit = {}
                    axis_obj = joint_data.get("axis", {})
                    if isinstance(axis_obj, dict):
                        axis_vec = axis_obj.get("direction", [0, 0, 1])
                    elif isinstance(axis_obj, list):
                        axis_vec = axis_obj
                    else:
                        axis_vec = [0, 0, 1]
                    # PartNet-Mobility quirk: revolute / continuous joint
                    # limits are stored in DEGREES, prismatic in meters. SAPIEN
                    # expects radians for revolute, so convert here.
                    a_raw = float(limit.get("a", 0.0))
                    b_raw = float(limit.get("b", 0.0))
                    if jtype in ("revolute", "continuous"):
                        a_val = math.radians(a_raw)
                        b_val = math.radians(b_raw)
                    else:
                        a_val = a_raw
                        b_val = b_raw
                    joints.append(
                        JointInfo(
                            name=str(entry.get("name", "")),
                            joint_type=jtype,
                            limit_low=a_val,
                            limit_high=b_val,
                            axis=list(axis_vec),
                            parent_link=str(entry.get("parent", "")),
                            child_link=str(entry.get("name", "")),
                        )
                    )

            has_textures = (folder / "textured_objs").exists()
            bbox_diag = _estimate_bbox_diag(folder) if compute_bbox else 1.0

            return cls(
                obj_id=folder.name,
                folder=folder,
                category=category,
                joints=joints,
                bbox_diagonal=bbox_diag,
                has_textures=has_textures,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping %s: %s", folder, e)
            return None

    # ------------------------------------------------------------ active SAPIEN
    def load(self, scene, fix_root: bool = True):
        """Load this object into an existing SAPIEN scene; returns the articulation.

        The caller owns the lifecycle.
        """
        sapien = _lazy_sapien()
        loader = scene.create_urdf_loader()
        loader.fix_root_link = fix_root
        articulation = loader.load(str(self.folder / "mobility.urdf"))
        if articulation is None:
            raise RuntimeError(f"SAPIEN failed to load {self.folder}")
        return articulation

    # ------------------------------------------------------------ utility
    @property
    def has_active_joint(self) -> bool:
        return any(j.is_active for j in self.joints)

    def largest_joint(self) -> Optional[JointInfo]:
        active = [j for j in self.joints if j.is_active]
        if not active:
            return None
        return max(active, key=lambda j: j.range)


# ----------------------------------------------------------------------------
# bbox estimate without spinning up SAPIEN: read URDF mesh AABB cheaply
# ----------------------------------------------------------------------------
def _estimate_bbox_diag(folder: Path) -> float:
    """Best-effort bbox diagonal in meters from the URDF mesh files.

    Doesn't load SAPIEN. Reads meshes via trimesh if available; falls back to 1.0.
    """
    try:
        import trimesh  # type: ignore[import]
    except ImportError:
        return 1.0

    mesh_dir = folder / "textured_objs"
    if not mesh_dir.exists():
        return 1.0

    mins, maxs = None, None
    for mesh_path in mesh_dir.glob("*.obj"):
        try:
            m = trimesh.load(mesh_path, force="mesh")
            if not hasattr(m, "vertices"):
                continue
            v = m.vertices
            cur_min, cur_max = v.min(axis=0), v.max(axis=0)
            if mins is None:
                mins, maxs = cur_min, cur_max
            else:
                mins = list(map(min, mins, cur_min))
                maxs = list(map(max, maxs, cur_max))
        except Exception:  # noqa: BLE001
            continue

    if mins is None or maxs is None:
        return 1.0
    diag = sum((b - a) ** 2 for a, b in zip(mins, maxs)) ** 0.5
    return float(diag)


# ----------------------------------------------------------------------------
# enumeration
# ----------------------------------------------------------------------------
def enumerate_partnet_objects(
    partnet_root: str | os.PathLike,
    only_categories: Optional[List[str]] = None,
    *,
    compute_bbox: bool = False,
) -> Iterator[PartNetObject]:
    """Yield every well-formed PartNetObject under <partnet_root>/dataset/.

    Args:
        partnet_root: path to your local PartNet-Mobility extraction.
        only_categories: lowercase category names (`meta.json["model_cat"]`).
                         If given, others are skipped.
        compute_bbox:    if True, load all mesh files via trimesh to compute
                         bbox diagonals. SLOW (~10x slower). Default False;
                         downstream code that actually needs bbox should compute
                         it lazily for the small filtered set.
    """
    partnet_root = Path(partnet_root).expanduser().resolve()
    dataset_dir = partnet_root / "dataset"
    if not dataset_dir.exists():
        # Some users keep the objects directly under the root
        dataset_dir = partnet_root

    if not dataset_dir.exists():
        raise FileNotFoundError(f"PartNet-Mobility root not found: {partnet_root}")

    only = {c.lower() for c in only_categories} if only_categories else None

    for child in sorted(dataset_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        # Quick category filter BEFORE expensive parsing
        if only is not None:
            try:
                with open(child / "meta.json") as f:
                    meta_quick = json.load(f)
                cat_quick = (meta_quick.get("model_cat") or meta_quick.get("category") or "").lower()
                if cat_quick not in only:
                    continue
            except Exception:
                continue

        obj = PartNetObject.from_folder(child, compute_bbox=compute_bbox)
        if obj is None:
            continue
        if only is not None and obj.category.lower() not in only:
            continue
        yield obj
