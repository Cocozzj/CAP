"""URDF parsing + minimal forward-kinematics for PartNet-Mobility objects.

For our evaluation we only need *one* thing per object: the geometry of
the **target joint** specified by ``meta.json["joint_index"]`` —
specifically its axis, origin, and type (revolute / prismatic).

We don't need the full kinematic tree because:
  • PartNet-Mobility ``mobility.urdf`` files are simple — usually 1 movable
    joint + 1 fixed root joint (see e.g. Box 100191).
  • ``trajectory.npz["joint_qpos"]`` is already 1-D scalar, containing only
    the target joint's evolution (the dataset converter pre-extracted it).
  • For ADE/FDE/MPJPE on the moving part, all we need is enough info to
    compute, given a joint angle θ, the rigid transform applied to the
    moving link's points relative to the parent.

This module avoids depending on ``yourdfpy`` / ``urdf_parser_py`` so it
runs in any conda env — we parse the URDF XML directly with
``xml.etree.ElementTree``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xml.etree.ElementTree as ET


@dataclass
class JointInfo:
    """The fields we actually need for FK on PartNet objects."""
    name:         str
    type:         str           # 'revolute' / 'prismatic' / 'fixed' / 'continuous'
    axis:         np.ndarray    # [3] unit vector in parent link's frame
    origin_xyz:   np.ndarray    # [3] joint origin in parent link's frame
    origin_rpy:   np.ndarray    # [3] joint orientation (Euler) in parent frame
    parent_link:  str
    child_link:   str
    limit_lower:  Optional[float] = None
    limit_upper:  Optional[float] = None


def _parse_xyz(s: str) -> np.ndarray:
    """'0 1 -2' → np.array([0, 1, -2])"""
    return np.array([float(v) for v in s.strip().split()], dtype=np.float32)


def parse_urdf(urdf_path: str | Path) -> Dict[str, JointInfo]:
    """Read a URDF file and return ``{joint_name: JointInfo}``.

    Robust to PartNet's quirky URDFs — they sometimes have multiple
    ``<visual>`` blocks per link, etc.  We only look at ``<joint>`` tags.
    """
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints: Dict[str, JointInfo] = {}
    for j in root.findall("joint"):
        name = j.attrib["name"]
        jtype = j.attrib.get("type", "fixed")
        # axis (default = [1, 0, 0] per URDF spec)
        axis_el = j.find("axis")
        axis = _parse_xyz(axis_el.attrib["xyz"]) if axis_el is not None \
               else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        # origin
        ori_el = j.find("origin")
        if ori_el is not None:
            xyz = _parse_xyz(ori_el.attrib.get("xyz", "0 0 0"))
            rpy = _parse_xyz(ori_el.attrib.get("rpy", "0 0 0"))
        else:
            xyz = np.zeros(3, dtype=np.float32)
            rpy = np.zeros(3, dtype=np.float32)
        # parent / child
        parent = j.find("parent").attrib.get("link", "")
        child  = j.find("child").attrib.get("link", "")
        # limits (only meaningful for revolute/prismatic)
        lim_el = j.find("limit")
        lo = float(lim_el.attrib["lower"]) if lim_el is not None and "lower" in lim_el.attrib else None
        hi = float(lim_el.attrib["upper"]) if lim_el is not None and "upper" in lim_el.attrib else None
        joints[name] = JointInfo(
            name=name, type=jtype, axis=axis,
            origin_xyz=xyz, origin_rpy=rpy,
            parent_link=parent, child_link=child,
            limit_lower=lo, limit_upper=hi,
        )
    return joints


def get_movable_joints(joints: Dict[str, JointInfo]) -> List[JointInfo]:
    """Return joints whose ``type`` is revolute / prismatic / continuous."""
    return [j for j in joints.values()
            if j.type in ("revolute", "prismatic", "continuous")]


def get_joint_by_index(joints: Dict[str, JointInfo],
                        idx: int) -> Optional[JointInfo]:
    """meta.json's ``joint_index`` numbers the *movable* joints in their
    URDF declaration order.  joint_index=0 → first movable joint, etc.
    """
    movable = sorted(get_movable_joints(joints), key=lambda j: j.name)
    if idx < 0 or idx >= len(movable):
        return None
    return movable[idx]


# ════════════════════════════════════════════════════════════════════════
# Forward-kinematics helpers
# ════════════════════════════════════════════════════════════════════════

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' formula — rotation by ``angle`` (radians) around ``axis``."""
    a = axis / max(float(np.linalg.norm(axis)), 1e-12)
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    x, y, z = a
    return np.array([
        [c + x*x*C,    x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,  c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,  z*y*C + x*s, c + z*z*C  ],
    ], dtype=np.float32)


def joint_pose(j: JointInfo, q: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the (R, t) transform applied to the *child link* relative
    to the *parent link*'s origin, given joint position ``q``:

      • revolute / continuous: rotation by ``q`` rad around ``j.axis``,
        about ``j.origin_xyz`` (after the joint's static rpy).
      • prismatic: translation by ``q`` along ``j.axis``.
      • fixed: identity rotation, fixed translation = ``j.origin_xyz``.

    Returns (R [3,3], t [3]) such that:
      child_pt_in_parent = R @ child_pt_in_child + t
    """
    # Joint static origin (always applied even for revolute, since it
    # places the joint frame inside the parent link).
    R_static = _rpy_to_R(j.origin_rpy)
    t_static = j.origin_xyz.astype(np.float32)

    if j.type in ("revolute", "continuous"):
        R_motion = rotation_matrix(j.axis, q)
        # Composed: parent frame → static origin → rotate
        R = R_static @ R_motion
        # Rotation is about the joint origin (already shifted by t_static).
        # In parent frame: x_parent = R_static @ (R_motion @ x_child) + t_static
        return R, t_static
    elif j.type == "prismatic":
        # Translation along axis (in joint's frame)
        t_motion = j.axis * q
        R = R_static
        # Translation: in parent frame
        t = t_static + R_static @ t_motion.astype(np.float32)
        return R, t
    else:  # fixed
        return R_static, t_static


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (URDF convention: extrinsic XYZ) → 3x3 rotation."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════
# Convenience: load URDF for a PartNet object id
# ════════════════════════════════════════════════════════════════════════

PARTNET_RAW_DIR_DEFAULT = (
    "/home/zejun/CAP/dataset_gen/raw_data/partnet-mobility/dataset"
)


def load_partnet_urdf(
    obj_id:           str,
    partnet_raw_dir:  str | Path = PARTNET_RAW_DIR_DEFAULT,
) -> Optional[Dict[str, JointInfo]]:
    """Load mobility.urdf for a PartNet object by id.  Returns None if missing."""
    p = Path(partnet_raw_dir) / str(obj_id) / "mobility.urdf"
    if not p.exists():
        return None
    return parse_urdf(p)


__all__ = [
    "JointInfo",
    "parse_urdf",
    "get_movable_joints",
    "get_joint_by_index",
    "rotation_matrix",
    "joint_pose",
    "load_partnet_urdf",
]
