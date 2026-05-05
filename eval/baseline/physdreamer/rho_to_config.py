"""Map our 9-dim ρ tuple → PhysDreamer's physics config.

Our ρ (Option C, see model/executor/deform/rho_parser.py) layout:
    [0] E         Young's modulus   (Pa)
    [1] ν         Poisson ratio     (dimensionless, 0..0.499)
    [2] ρ_m       mass density      (kg/m^3)
    [3..5] F[3]   external force    (N, world-frame xyz)
    [6] μ         friction          (0..1)
    [7] damping   velocity damping  (0..1)
    [8] dt        per-step timestep (s)

PhysDreamer's expected MPM config (verify against actual source after clone):
    {
      "material_type": "elastic" | "plasticine" | "metal" | "jelly" | "sand",
      "E":             float,          # Young's modulus
      "nu":            float,          # Poisson ratio
      "rho":           float,          # density
      "yield_stress":  float,          # plastic yield (only for plasticine/metal)
      "external_force":[float, float, float],
      "friction":      float,
      "damping":       float,
      "dt":            float,
      "n_substeps":    int,            # MPM substeps per output frame
      "duration":      float,          # total sim time (s)
    }
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Material classification — map (E, ν, ρ_m) → PhysDreamer preset
# ──────────────────────────────────────────────────────────────────────
# PhysDreamer typically supports a few discrete material types; we pick the
# closest one based on our predicted (E, ν, ρ_m).  Thresholds calibrated
# against typical PartNet-Mobility object materials.

def classify_material(E: float, nu: float, rho_m: float) -> Dict:
    """Pick PhysDreamer material preset + overrides."""
    if E >= 1e10:
        # Very stiff (iron / metal furniture)
        return {"material_type": "metal", "yield_stress": 1e8}
    if E >= 1e8:
        # Stiff (rigid plastic, hard wood)
        return {"material_type": "plasticine", "yield_stress": 5e6}
    if E >= 1e6:
        # Medium stiffness (rubber, soft plastic)
        return {"material_type": "elastic", "yield_stress": None}
    if E >= 1e4:
        # Soft elastic (foam, rubber)
        return {"material_type": "jelly", "yield_stress": None}
    # Very soft / fluid-like
    return {"material_type": "jelly", "yield_stress": None}


def rho_to_physdreamer_config(
    rho:           Sequence[float],
    duration_secs: float,
    fps:           int = 30,
    n_substeps:    int = 100,
) -> Dict:
    """Convert our 9-dim ρ tuple → PhysDreamer config dict.

    Args:
      rho:           length-9 vector (see module docstring)
      duration_secs: total simulation time
      fps:           output frame rate
      n_substeps:    MPM substeps per output frame (controls stability)

    Returns a dict ready to dump to PhysDreamer's expected config JSON.
    """
    if len(rho) != 9:
        raise ValueError(f"ρ must have 9 dims (got {len(rho)})")

    E, nu, rho_m, fx, fy, fz, mu, damp, dt = [float(v) for v in rho]

    mat = classify_material(E, nu, rho_m)

    cfg: Dict = {
        "material_type":  mat["material_type"],
        "E":              max(E, 1e3),                           # PhysDreamer rejects E=0
        "nu":             float(np.clip(nu, 0.0, 0.499)),
        "rho":            max(rho_m, 1.0),
        "external_force": [fx, fy, fz],
        "friction":       float(np.clip(mu, 0.0, 1.0)),
        "damping":        float(np.clip(damp, 0.0, 1.0)),
        "dt":             max(dt, 1e-4),
        "n_substeps":     int(n_substeps),
        "duration":       float(duration_secs),
        "fps":            int(fps),
    }
    if mat.get("yield_stress") is not None:
        cfg["yield_stress"] = float(mat["yield_stress"])
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
        return [1e5, 0.40, 50.0,  0.0, 0.0, -9.81, 0.50, 0.10, 1.0/30.0]
    if cat in medium:
        return [1e8, 0.35, 800.0, 0.0, 0.0, -9.81, 0.40, 0.05, 1.0/30.0]
    return [2e11, 0.30, 7800.0,   0.0, 0.0, -9.81, 0.30, 0.05, 1.0/30.0]
