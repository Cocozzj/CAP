"""Map our 9-dim ρ tuple → PhysGaussian's actual config schema.

PhysGaussian's expected JSON schema (verified against config/ficus_config.json
in the official repo as of 2024-05):

    {
      "opacity_threshold": 0.02,
      "rotation_degree":   [0.0],
      "rotation_axis":     [0],

      "substep_dt": 1e-4,       MPM internal step
      "frame_dt":   1/fps,      output frame period
      "frame_num":  T           total frames

      "E":          float,      Young's modulus
      "nu":         float,      Poisson ratio
      "material":   "jelly"|"water"|"sand"|"metal"|"plasticine",
      "density":    float,
      "g":          [gx, gy, gz],          gravity vector
      "grid_v_damping_scale":  float ∈ (0, 1],  per-substep velocity scale
      "rpic_damping":          float,

      "boundary_conditions": [
        {"type": "cuboid", "point": [...], "size": [...], "velocity": [...],
         "start_time": 0, "end_time": 1e3, "reset": 1},        ← static walls
        {"type": "particle_impulse", "force": [fx, fy, fz],
         "num_dt": 1, "start_time": 0}                         ← external force
      ],
      "additional_material_params": [],

      "mpm_space_vertical_upward_axis": [0, 0, 1],
      "mpm_space_viewpoint_center":     [0, 0, 0],
      "default_camera_index":           -1,
      "show_hint":                      false,
      "init_azimuthm":  float, "init_elevation": float, "init_radius": float,
      "move_camera":   false,
      "delta_a": 0, "delta_e": 0, "delta_r": 0
    }

Our ρ tuple (Option C; see model/executor/deform/rho_parser.py):
    [0] E, [1] ν, [2] ρ_m, [3..5] external_force, [6] friction, [7] damping, [8] dt
"""
from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Material classification (PhysGaussian's enum)
# ──────────────────────────────────────────────────────────────────────

def classify_material(E: float, nu: float, rho_m: float) -> str:
    """Pick PhysGaussian's material preset based on stiffness."""
    if E >= 1e10:
        return "metal"          # very stiff
    if E >= 1e8:
        return "plasticine"     # plastic deformation
    if E >= 1e6:
        return "jelly"          # elastic soft
    return "jelly"              # default soft


# ──────────────────────────────────────────────────────────────────────
# Damping conversion
# ──────────────────────────────────────────────────────────────────────

def damping_to_grid_v_scale(damping_per_frame: float,
                              frame_dt: float, substep_dt: float) -> float:
    """Convert per-frame damping rate (∈ [0,1]) → per-substep velocity scale.

    PhysGaussian multiplies grid velocity by grid_v_damping_scale every substep.
    To match a per-frame damping ratio d:
        v_after_one_frame = v * (1 - d) = v * scale^(frame_dt / substep_dt)
        → scale = (1 - d)^(substep_dt / frame_dt)

    Returns scale in (0, 1].  Clamps for stability.
    """
    d = float(np.clip(damping_per_frame, 0.0, 0.99))
    if d <= 0.0:
        return 0.9999          # near-1 (almost no damping)
    n_substeps = max(frame_dt / max(substep_dt, 1e-9), 1.0)
    scale = (1.0 - d) ** (1.0 / n_substeps)
    return float(np.clip(scale, 0.5, 0.9999))


# ──────────────────────────────────────────────────────────────────────
# Main: ρ → PhysGaussian config
# ──────────────────────────────────────────────────────────────────────

def rho_to_physgaussian_config(
    rho:        Sequence[float],
    n_frames:   int,
    fps:        int   = 30,
    substep_dt: float = 1e-4,
    gravity:    Sequence[float] = (0.0, 0.0, -9.81),
) -> Dict[str, Any]:
    """Convert our 9-dim ρ tuple → PhysGaussian config dict (v2024.05 schema).

    Args:
      rho:        length-9 vector [E, ν, ρ_m, F[3], μ, damping, dt]
                  - F[3] = external force vector (NOT gravity; gravity is separate)
                  - dt   = our model's per-step dt; PhysGaussian gets its own
                           substep_dt (1e-4) + frame_dt (1/fps) instead
      n_frames:   total output frames (= T)
      fps:        output frame rate
      substep_dt: PhysGaussian MPM internal step (default 1e-4 = stable)
      gravity:    gravity vector (default Earth gravity along -z)
    """
    if len(rho) != 9:
        raise ValueError(f"ρ must have 9 dims (got {len(rho)})")

    E, nu, rho_m, fx, fy, fz, mu, damp, _dt_unused = [float(v) for v in rho]
    frame_dt = 1.0 / float(fps)

    cfg: Dict[str, Any] = {
        # ── Required: scene / opacity ──
        "opacity_threshold": 0.02,
        "rotation_degree":   [0.0],
        "rotation_axis":     [0],

        # ── Time stepping ──
        "substep_dt": float(substep_dt),
        "frame_dt":   float(frame_dt),
        "frame_num":  int(n_frames),

        # ── Material ──
        "E":          max(E, 1e3),                            # PhysGaussian rejects E≤0
        "nu":         float(np.clip(nu, 0.0, 0.499)),         # avoid incompressible 0.5
        "material":   classify_material(E, nu, rho_m),
        "density":    max(rho_m, 1.0),

        # ── Forces / damping ──
        "g":                       list(gravity),
        "grid_v_damping_scale":    damping_to_grid_v_scale(damp, frame_dt, substep_dt),
        "rpic_damping":            0.0,

        # ── Boundary conditions: external force from our F[3] ──
        # If our predicted ρ has a non-trivial external force, apply it as a
        # particle_impulse at t=0.  Friction is not a top-level field in
        # PhysGaussian — we approximate via grid_v_damping_scale.
        "boundary_conditions": [
            {
                "type":        "particle_impulse",
                "force":       [float(fx), float(fy), float(fz)],
                "num_dt":      1,
                "start_time":  0,
            }
        ] if (abs(fx) + abs(fy) + abs(fz) > 1e-6) else [],

        "additional_material_params": [],

        # ── Coordinate convention ──
        "mpm_space_vertical_upward_axis": [0, 0, 1],
        "mpm_space_viewpoint_center":     [0.0, 0.0, 0.0],

        # ── Camera (we don't render with PhysGaussian's renderer here) ──
        "default_camera_index": -1,
        "show_hint":            False,
        "init_azimuthm":        0.0,
        "init_elevation":       0.0,
        "init_radius":          4.0,
        "move_camera":          False,
        "delta_a":              0.0,
        "delta_e":              0.0,
        "delta_r":              0.0,
    }
    return cfg


def default_rho_for_partnet_object(obj_category: str | None) -> Sequence[float]:
    """Fallback ρ when no GT physics params exist (e.g. Dataset-B real video).

    Heuristic mapping object_category → typical material:
      Cloth / SoftToy            → soft jelly-like
      Box / Suitcase             → medium plasticine
      Refrigerator / Microwave / Oven / Dishwasher / Door / Window / Faucet
        / StorageFurniture_*     → metal (rigid)
      everything else            → metal (default)
    """
    cat = (obj_category or "").lower()
    soft = {"cloth", "softtoy"}
    medium = {"box", "suitcase", "kettle"}
    if cat in soft:
        return [1e5, 0.40, 50.0,  0.0, 0.0, 0.0, 0.50, 0.10, 1.0/30.0]
    if cat in medium:
        return [1e8, 0.35, 800.0, 0.0, 0.0, 0.0, 0.40, 0.05, 1.0/30.0]
    return [2e11, 0.30, 7800.0,   0.0, 0.0, 0.0, 0.30, 0.05, 1.0/30.0]
