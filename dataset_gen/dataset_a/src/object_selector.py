"""Pick the final object set: combine PartNet enumeration + category whitelist
+ motion-saliency filter + per-category cap.

Used by `scripts/01_filter_objects.py`.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from tqdm import tqdm

from .motion_saliency import (
    SaliencyScore,
    make_quick_renderer,
    score_object,
)
from .object_loader import (
    PartNetObject,
    enumerate_partnet_objects,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# config helpers
# ----------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def whitelisted_partnet_categories(category_cfg: dict) -> List[str]:
    """Lowercase PartNet model_cat names from object_categories.yaml."""
    return [c["partnet_category_name"] for c in category_cfg["categories"]]


def category_rules(category_cfg: dict) -> Dict[str, List[dict]]:
    """Group multiple of our category specs by their PartNet model_cat name.

    Some PartNet categories (e.g. 'storagefurniture') host both doors and
    drawers — we have *two* of our category entries pointing at the same
    PartNet name. This returns {partnet_name: [our_spec, ...]}.

    Procedural-source categories (no PartNet asset) are excluded.
    """
    out: Dict[str, List[dict]] = defaultdict(list)
    for c in category_cfg["categories"]:
        if c.get("source") == "procedural":
            continue
        partnet_name = c.get("partnet_category_name")
        if not partnet_name:
            continue
        out[partnet_name.lower()].append(c)
    return out


def first_match_for_object(obj: PartNetObject, rules: Dict[str, List[dict]]) -> Optional[dict]:
    """Choose which of our category specs (if any) accepts this object based
    on its joint inventory. Returns the matching spec dict, or None."""
    candidates = rules.get(obj.category.lower(), [])
    if not candidates:
        return None
    for spec in candidates:
        wanted_type = spec.get("joint_type", "any")
        min_range = spec.get("min_joint_range", 0.0)
        for j in obj.joints:
            if not j.is_active:
                continue
            if wanted_type != "any" and wanted_type != "mixed" and j.joint_type != wanted_type:
                continue
            if abs(j.range) < min_range:
                continue
            # Found at least one usable joint of the right kind
            return spec
    return None


# ----------------------------------------------------------------------------
# top-level filter
# ----------------------------------------------------------------------------
def expand_procedural_categories(category_cfg: dict, *, seed: int = 0) -> List[dict]:
    """For categories with `source: procedural`, generate the synthetic instance specs.

    Returns a list of object records in the same shape as PartNet-derived ones,
    minus the joint info (filled with sentinels).
    """
    from .soft_objects import enumerate_soft_instances

    out = []
    for c in category_cfg["categories"]:
        if c.get("source") != "procedural":
            continue
        cat_name = c["name"]
        n = c.get("n_instances", 10)
        specs = enumerate_soft_instances(cat_name, n_instances=n, seed=seed)
        for i, spec in enumerate(specs):
            out.append({
                "obj_id": f"proc_{cat_name}_{i:03d}",
                "partnet_category": cat_name.lower(),
                "our_category": cat_name,
                "folder": "",                          # no PartNet folder
                "joint_index": -1,
                "joint_name": "",
                "joint_type": "none",
                "joint_range": 0.0,
                "joint_limit_low": 0.0,
                "joint_limit_high": 0.0,
                "bbox_diagonal": spec.size * 1.5,      # rough
                "soft_object_spec": spec.to_dict(),
                "is_procedural": True,
                "saliency": {
                    "obj_id": f"proc_{cat_name}_{i:03d}",
                    "category": cat_name,
                    "joint_index": -1,
                    "geometry_score": 1.0,
                    "volume_score": 1.0,
                    "pixel_score": 1.0,
                    "total": 100.0,
                },
            })
    return out


def select_objects(
    partnet_root: str | Path,
    category_cfg: dict,
    saliency_cfg: dict,
    instances_per_category: int,
    *,
    verbose: bool = True,
) -> List[dict]:
    """Run the full filter pipeline; returns a JSON-serializable list of
    selected object records.

    Each record:
      {
        "obj_id": "100147",
        "partnet_category": "door",
        "our_category": "Door",
        "folder": "/abs/path/.../100147",
        "joint_index": 0,
        "joint_type": "revolute",
        "joint_range": 1.7,
        "bbox_diagonal": 1.2,
        "saliency": {...},   # full SaliencyScore.__dict__
      }
    """
    rules = category_rules(category_cfg)
    whitelist = list(rules.keys())

    # Phase 1: enumerate + per-category match
    matched: Dict[str, List[tuple[PartNetObject, dict]]] = defaultdict(list)
    pbar = tqdm(
        enumerate_partnet_objects(partnet_root, only_categories=whitelist),
        desc="Enumerating PartNet-Mobility",
        disable=not verbose,
    )
    for obj in pbar:
        spec = first_match_for_object(obj, rules)
        if spec is None:
            continue
        matched[spec["name"]].append((obj, spec))

    if verbose:
        for name, items in matched.items():
            logger.info("Category %s: %d candidates after geometry filter", name, len(items))

    # Phase 2: motion-saliency rendering filter
    selected: List[dict] = []
    if saliency_cfg.get("enable", True):
        renderer = make_quick_renderer(image_size=128)
    else:
        renderer = None

    for our_cat, items in matched.items():
        scored: List[tuple[PartNetObject, dict, SaliencyScore]] = []
        for obj, spec in tqdm(items, desc=f"Saliency: {our_cat}", disable=not verbose):
            if renderer is None:
                # No saliency filter: pick the LARGEST-range matching joint
                # (NOT joint[0], which in URDF is often a tiny button/knob).
                wanted_type = spec.get("joint_type", "any")
                only_idx = spec.get("only_joint_index")
                # Joint name blacklist (case-insensitive substring match): skip known
                # problematic joint names like locked door panels or fixed parts.
                blacklist = [s.lower() for s in spec.get("joint_name_blacklist", [
                    "surface_board", "fixed_part", "fixed", "base_link",
                    "frame", "lock", "mirror",
                ])]
                best_idx = None
                best_range = 0.0
                for i, j in enumerate(obj.joints):
                    if not j.is_active:
                        continue
                    if only_idx is not None and i != only_idx:
                        continue
                    if wanted_type not in ("any", "mixed") and j.joint_type != wanted_type:
                        continue
                    jname_lower = j.name.lower() if j.name else ""
                    if any(b in jname_lower for b in blacklist):
                        continue
                    if abs(j.range) > best_range:
                        best_range = abs(j.range)
                        best_idx = i
                if best_idx is None:
                    continue
                score = SaliencyScore(
                    obj_id=obj.obj_id, category=our_cat,
                    joint_index=best_idx,
                    geometry_score=0, volume_score=0,
                    pixel_score=float(best_range),
                    total=float(best_range),
                )
                scored.append((obj, spec, score))
                continue

            score = score_object(obj, renderer=renderer)
            if score is None:
                continue
            if not score.passes(saliency_cfg):
                continue
            scored.append((obj, spec, score))

        # Rank by total score, take top instances_per_category
        scored.sort(key=lambda x: x[2].total, reverse=True)
        keep = scored[: instances_per_category]
        for obj, spec, score in keep:
            joint = obj.joints[score.joint_index]
            selected.append(
                {
                    "obj_id": obj.obj_id,
                    "partnet_category": obj.category,
                    "our_category": our_cat,
                    "folder": str(obj.folder),
                    "joint_index": score.joint_index,
                    "joint_name": joint.name,
                    "joint_type": joint.joint_type,
                    "joint_range": float(joint.range),
                    "joint_limit_low": float(joint.limit_low),
                    "joint_limit_high": float(joint.limit_high),
                    "bbox_diagonal": obj.bbox_diagonal,
                    "saliency": asdict(score),
                }
            )

    # Append procedural soft categories (don't go through PartNet enumeration)
    proc = expand_procedural_categories(category_cfg)
    if proc:
        selected.extend(proc)
        if verbose:
            logger.info("Added %d procedural (soft) instances", len(proc))

    if verbose:
        per_cat = defaultdict(int)
        for s in selected:
            per_cat[s["our_category"]] += 1
        for cat, n in sorted(per_cat.items()):
            logger.info("  Selected %d for %s", n, cat)

    return selected


def save_object_list(selected: List[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"objects": selected, "n": len(selected)}, f, indent=2)
    logger.info("Wrote %d objects to %s", len(selected), out_path)
