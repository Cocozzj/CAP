"""Procedurally-generated deformable objects (no PartNet asset required).

We use procedural primitives (cube, sphere, planar grid) instead of pulling
from PartNet-Mobility because PartNet has very few soft / deformable items.
For training-data purposes a clean parametric mesh is fine — the model only
sees the rendered RGB.

Provides:
  make_soft_cube(size, color, n_subdivisions)        -> trimesh.Trimesh
  make_planar_cloth(width, height, color, divisions) -> trimesh.Trimesh
  apply_anisotropic_scale(mesh, scale_xyz)           -> trimesh.Trimesh
  apply_hinge_fold(mesh, hinge_axis, hinge_origin, angle_rad) -> trimesh.Trimesh

The deformation functions return new trimesh objects (do not mutate input).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class SoftObjectSpec:
    """Self-contained recipe for one soft-object instance.

    Persisted in the trajectory record so the renderer can recreate the rest
    mesh deterministically from the spec without needing to ship a .obj file.
    """
    primitive: str              # "soft_cube" | "soft_sphere" | "cloth"
    size: float                 # characteristic length (m)
    color: list                 # [r, g, b] in [0, 1]
    n_subdivisions: int = 8     # mesh resolution
    aspect: float = 1.0         # for cloth: width/height ratio
    seed: int = 0               # for any randomness in mesh generation

    def to_dict(self) -> dict:
        return {
            "primitive": self.primitive,
            "size": self.size,
            "color": self.color,
            "n_subdivisions": self.n_subdivisions,
            "aspect": self.aspect,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SoftObjectSpec":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------------
# primitive generators
# ----------------------------------------------------------------------------
def make_soft_cube(size: float = 0.3,
                    color: Sequence[float] = (0.7, 0.5, 0.5),
                    n_subdivisions: int = 8):
    """A subdivided cube; vertices are dense enough that anisotropic scaling
    doesn't look too low-poly. Centered at origin."""
    import trimesh
    box = trimesh.creation.box(extents=(size, size, size))
    # Subdivide for smoother deformation
    for _ in range(int(math.log2(max(n_subdivisions, 1)))):
        box = box.subdivide()
    box.visual.face_colors = [int(c * 255) for c in color] + [255]
    return box


def make_soft_sphere(size: float = 0.3,
                      color: Sequence[float] = (0.7, 0.5, 0.5),
                      n_subdivisions: int = 3):
    """An icosphere; smooth deformation under scaling."""
    import trimesh
    sph = trimesh.creation.icosphere(subdivisions=n_subdivisions, radius=size / 2)
    sph.visual.face_colors = [int(c * 255) for c in color] + [255]
    return sph


def make_planar_cloth(width: float = 0.5,
                       height: float = 0.4,
                       color: Sequence[float] = (0.4, 0.5, 0.7),
                       n_subdivisions: int = 16):
    """A planar grid mesh, lying in the XZ plane, centered at origin.

    For folding, the natural hinge is the X axis (line through Y=0 in the plane),
    so we orient the cloth so its long dimension is along Z.
    """
    import trimesh
    # Build a regular grid mesh in the XY plane, then rotate to XZ
    nx = max(2, n_subdivisions)
    ny = max(2, int(n_subdivisions * height / width))
    xs = np.linspace(-width / 2, width / 2, nx)
    ys = np.linspace(-height / 2, height / 2, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    verts = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3)

    # Triangle faces over the grid
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            v0 = i * ny + j
            v1 = (i + 1) * ny + j
            v2 = (i + 1) * ny + (j + 1)
            v3 = i * ny + (j + 1)
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    faces = np.array(faces, dtype=np.int64)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # Rotate so the cloth lies in XZ (Y becomes the "up" of folding)
    R = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
    mesh.apply_transform(R)

    mesh.visual.face_colors = [int(c * 255) for c in color] + [255]
    return mesh


def make_from_spec(spec: SoftObjectSpec):
    """Dispatch to the right primitive constructor."""
    if spec.primitive == "soft_cube":
        return make_soft_cube(size=spec.size, color=spec.color,
                              n_subdivisions=spec.n_subdivisions)
    if spec.primitive == "soft_sphere":
        return make_soft_sphere(size=spec.size, color=spec.color,
                                n_subdivisions=min(spec.n_subdivisions, 4))
    if spec.primitive == "cloth":
        return make_planar_cloth(
            width=spec.size,
            height=spec.size * spec.aspect,
            color=spec.color,
            n_subdivisions=spec.n_subdivisions,
        )
    raise ValueError(f"Unknown primitive: {spec.primitive!r}")


# ----------------------------------------------------------------------------
# deformations (return new trimesh)
# ----------------------------------------------------------------------------
def apply_anisotropic_scale(mesh, scale_xyz: Sequence[float]):
    """Scale vertices along world XYZ. Use small scale (e.g. 0.6) to "squeeze"."""
    import trimesh
    sx, sy, sz = scale_xyz
    new_verts = mesh.vertices * np.array([sx, sy, sz], dtype=np.float64)[None, :]
    out = trimesh.Trimesh(vertices=new_verts, faces=mesh.faces.copy(), process=False)
    if hasattr(mesh, "visual") and getattr(mesh.visual, "face_colors", None) is not None:
        out.visual.face_colors = mesh.visual.face_colors.copy()
    return out


def apply_hinge_fold(mesh,
                      hinge_axis: Sequence[float] = (1, 0, 0),
                      hinge_origin: Sequence[float] = (0, 0, 0),
                      fold_angle_rad: float = 0.0,
                      side_to_fold: str = "positive"):
    """Bend the mesh around a hinge line.

    Vertices on the `side_to_fold` of the hinge plane are rotated by
    `fold_angle_rad` around the axis through `hinge_origin` along `hinge_axis`.
    Vertices on the other side stay fixed.

    For a planar cloth in XZ plane folded around the X axis: the "positive Z"
    half rotates while the "negative Z" half stays.
    """
    import trimesh
    axis = np.array(hinge_axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    origin = np.array(hinge_origin, dtype=np.float64)

    verts = mesh.vertices.copy()

    # Pick the perpendicular direction in mesh's natural up-axis
    # For a cloth in XZ plane folded around X axis: split by Z sign
    # Generic approach: compute perpendicular component and pick by sign
    # For our use case (cloth in XZ, hinge along X), we partition by Z.
    # Generalize: project verts onto a perpendicular axis to the hinge.
    if abs(axis[0]) > 0.99:
        perp = np.array([0, 0, 1.0])
    elif abs(axis[2]) > 0.99:
        perp = np.array([0, 1.0, 0])
    else:
        perp = np.array([0, 1.0, 0])

    perp_coord = (verts - origin) @ perp
    if side_to_fold == "positive":
        mask = perp_coord > 1e-6
    else:
        mask = perp_coord < -1e-6

    # Rotation matrix around axis
    c = math.cos(fold_angle_rad)
    s = math.sin(fold_angle_rad)
    K = np.array([
        [0,        -axis[2],  axis[1]],
        [axis[2],   0,       -axis[0]],
        [-axis[1],  axis[0],  0      ],
    ])
    R = np.eye(3) + s * K + (1 - c) * (K @ K)

    # Rotate selected vertices around (hinge_origin)
    moving = verts[mask] - origin
    moving = moving @ R.T
    moving = moving + origin
    verts[mask] = moving

    out = trimesh.Trimesh(vertices=verts, faces=mesh.faces.copy(), process=False)
    if hasattr(mesh, "visual") and getattr(mesh.visual, "face_colors", None) is not None:
        out.visual.face_colors = mesh.visual.face_colors.copy()
    return out


# ----------------------------------------------------------------------------
# instance enumeration: produce a deterministic set of soft objects
# ----------------------------------------------------------------------------
def enumerate_soft_instances(category: str, n_instances: int, seed: int = 0):
    """Return a list of SoftObjectSpec deterministically randomized per instance.

    Used by `object_selector` to expand the soft categories into concrete
    instance specs, the same way PartNet enumerates physical instances.
    """
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_instances):
        if category in ("SoftToy", "Sponge"):
            primitive = rng.choice(["soft_cube", "soft_sphere"])
            size = float(rng.uniform(0.20, 0.35))
            color = rng.uniform(0.3, 0.9, size=3).tolist()
            spec = SoftObjectSpec(
                primitive=str(primitive),
                size=size,
                color=color,
                n_subdivisions=8,
                aspect=1.0,
                seed=int(i),
            )
        elif category in ("Cloth", "Towel"):
            size = float(rng.uniform(0.40, 0.70))
            aspect = float(rng.uniform(0.6, 1.0))
            color = rng.uniform(0.3, 0.9, size=3).tolist()
            spec = SoftObjectSpec(
                primitive="cloth",
                size=size,
                color=color,
                n_subdivisions=16,
                aspect=aspect,
                seed=int(i),
            )
        else:
            raise ValueError(f"Unknown soft category: {category!r}")
        out.append(spec)
    return out
