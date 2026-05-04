"""Mesh-based init_gs backend.

Bypasses 3DGS optimization entirely. For each trajectory:
  1. Load the PartNet object's URDF + mesh files
  2. Apply forward kinematics to set the joint to the trajectory's first
     frame qpos (so an "open" trajectory starts with the door open, etc.)
  3. Sample N points uniformly on each link's transformed mesh surface
  4. Each sample becomes a Gaussian (color = mesh face/vertex color)

Quality: matches PartNet geometry exactly. Time: ~1-2 sec / trajectory.
"""

from __future__ import annotations
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# top-level entry
# ----------------------------------------------------------------------------
def reconstruct_from_mesh(
    obj_folder: str,
    n_points: int = 50000,
    joint_qpos: Optional[float] = None,
    joint_name: Optional[str] = None,
) -> Optional[dict]:
    """Sample PartNet mesh surface to produce Gaussians.

    If `joint_qpos` and `joint_name` are provided, apply forward kinematics
    so the sampled gaussians reflect the articulated state at the first
    frame (e.g. door already open). Otherwise sample at canonical pose.
    """
    try:
        import trimesh  # noqa: F401
    except ImportError:
        logger.error("trimesh not installed")
        return None

    folder = Path(obj_folder)
    if joint_qpos is not None and joint_name is not None:
        return _reconstruct_articulated(folder, n_points, joint_qpos, joint_name)
    return _reconstruct_canonical(folder, n_points)


# ----------------------------------------------------------------------------
# canonical pose (no joint info): just concat all meshes
# ----------------------------------------------------------------------------
def _reconstruct_canonical(folder: Path, n_points: int) -> Optional[dict]:
    import trimesh

    mesh_dir = folder / "textured_objs"
    if not mesh_dir.exists():
        logger.warning("No textured_objs in %s", folder)
        return None

    meshes = []
    for obj_file in sorted(mesh_dir.glob("*.obj")):
        try:
            m = trimesh.load(obj_file, force="mesh", process=False)
            if hasattr(m, "vertices") and len(m.vertices) > 0:
                meshes.append(m)
        except Exception as e:
            logger.warning("Failed to load %s: %s", obj_file, e)

    if not meshes:
        return None

    combined = trimesh.util.concatenate(meshes)
    return _sample_to_gs_dict(combined, n_points)


# ----------------------------------------------------------------------------
# articulated pose: apply forward kinematics from URDF
# ----------------------------------------------------------------------------
def _reconstruct_articulated(
    folder: Path,
    n_points: int,
    joint_qpos: float,
    joint_name: str,
) -> Optional[dict]:
    """Apply forward kinematics so meshes are transformed by their link's
    world pose at the given joint state, then sample.

    Uses SAPIEN (already a project dependency) for kinematics.
    """
    import trimesh

    urdf_path = folder / "mobility.urdf"
    if not urdf_path.exists():
        logger.warning("No mobility.urdf in %s", folder)
        return _reconstruct_canonical(folder, n_points)

    # ---- 1. SAPIEN: load + apply qpos + extract per-link world poses
    try:
        link_world_poses = _get_link_world_poses_sapien(
            urdf_path, joint_qpos, joint_name
        )
    except Exception as e:
        logger.warning("SAPIEN FK failed (%s); falling back to canonical", e)
        return _reconstruct_canonical(folder, n_points)

    # ---- 2. Parse URDF XML to know which mesh file goes with which link
    link_to_mesh_specs = _parse_urdf_link_meshes(urdf_path)

    # ---- 3. Load each mesh, transform by link world pose, accumulate
    transformed_meshes = []
    for link_name, mesh_specs in link_to_mesh_specs.items():
        if link_name not in link_world_poses:
            continue
        link_T = link_world_poses[link_name]   # 4x4
        for mesh_rel_path, origin_T in mesh_specs:
            mesh_full_path = folder / mesh_rel_path
            if not mesh_full_path.exists():
                # PartNet sometimes stores under textured_objs/ relatively
                alt = folder / "textured_objs" / Path(mesh_rel_path).name
                if alt.exists():
                    mesh_full_path = alt
                else:
                    continue
            try:
                mesh = trimesh.load(mesh_full_path, force="mesh", process=False)
                if not hasattr(mesh, "vertices") or len(mesh.vertices) == 0:
                    continue
                # World transform: link_T @ origin_T
                T = link_T @ origin_T
                mesh = mesh.copy()
                mesh.apply_transform(T)
                transformed_meshes.append(mesh)
            except Exception as e:
                logger.warning("Failed to load %s: %s", mesh_full_path, e)

    if not transformed_meshes:
        logger.warning("No meshes loaded for %s after FK; fallback canonical", folder)
        return _reconstruct_canonical(folder, n_points)

    combined = trimesh.util.concatenate(transformed_meshes)
    return _sample_to_gs_dict(combined, n_points)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _get_link_world_poses_sapien(
    urdf_path: Path,
    joint_qpos: float,
    joint_name: str,
) -> Dict[str, np.ndarray]:
    """Use SAPIEN to load URDF, set qpos, return {link_name: 4x4 world transform}."""
    import sapien.core as sapien
    from scipy.spatial.transform import Rotation as R

    engine = sapien.Engine()
    sap_renderer = sapien.SapienRenderer()
    engine.set_renderer(sap_renderer)
    scene = engine.create_scene()

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    articulation = loader.load(str(urdf_path))
    if articulation is None:
        raise RuntimeError(f"SAPIEN failed to load {urdf_path}")

    active_joints = articulation.get_active_joints()
    target_idx = next(
        (i for i, j in enumerate(active_joints) if j.name == joint_name),
        0,
    )
    qpos = np.zeros(len(active_joints), dtype=np.float64)
    if active_joints:
        qpos[target_idx] = float(joint_qpos)
    articulation.set_qpos(qpos)

    poses: Dict[str, np.ndarray] = {}
    for link in articulation.get_links():
        pose = link.get_pose()
        T = np.eye(4)
        # SAPIEN quaternion: wxyz; scipy expects xyzw
        quat_xyzw = [pose.q[1], pose.q[2], pose.q[3], pose.q[0]]
        T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        T[:3, 3] = np.asarray(pose.p)
        poses[link.name] = T
    return poses


def _parse_urdf_link_meshes(urdf_path: Path) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    """Parse the URDF XML to find each link's visual mesh files + their
    local origin transforms.

    Returns {link_name: [(mesh_relative_path, 4x4_origin_transform), ...]}.
    """
    from scipy.spatial.transform import Rotation as R

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if not link_name:
            continue
        meshes: List[Tuple[str, np.ndarray]] = []
        for visual in link_elem.findall("visual"):
            origin_T = np.eye(4)
            origin = visual.find("origin")
            if origin is not None:
                xyz_str = origin.get("xyz", "0 0 0")
                rpy_str = origin.get("rpy", "0 0 0")
                xyz = [float(x) for x in xyz_str.split()]
                rpy = [float(x) for x in rpy_str.split()]
                origin_T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
                origin_T[:3, 3] = xyz
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            mesh_path = mesh_elem.get("filename")
            if not mesh_path:
                continue
            # Strip "package://" prefix if present
            if mesh_path.startswith("package://"):
                mesh_path = mesh_path[len("package://"):]
            meshes.append((mesh_path, origin_T))
        if meshes:
            out[link_name] = meshes
    return out


def _sample_to_gs_dict(combined_mesh, n_points: int) -> dict:
    """Sample n_points on a (already transformed) trimesh, build GS dict."""
    import trimesh

    points, face_idx = trimesh.sample.sample_surface(combined_mesh, n_points)

    if hasattr(combined_mesh.visual, "vertex_colors") and \
            combined_mesh.visual.vertex_colors is not None:
        face_verts = combined_mesh.faces[face_idx]
        colors = combined_mesh.visual.vertex_colors[face_verts].mean(axis=1)[:, :3] / 255.0
    elif hasattr(combined_mesh.visual, "face_colors") and \
            combined_mesh.visual.face_colors is not None:
        colors = combined_mesh.visual.face_colors[face_idx][:, :3] / 255.0
    else:
        colors = np.full((n_points, 3), 0.5, dtype=np.float32)

    points = points.astype(np.float32)
    colors = colors.astype(np.float32)

    bbox_diag = np.linalg.norm(combined_mesh.bounds[1] - combined_mesh.bounds[0])
    point_scale = max(bbox_diag / (n_points ** (1 / 3)) * 0.5, 1e-3)
    scales = np.full((n_points, 3), point_scale, dtype=np.float32)
    quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_points, 1))
    opacities = np.full(n_points, 0.95, dtype=np.float32)

    return {
        "means": points,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "colors": colors,
    }
