"""Map our 9-dim ρ tuple → PhysGaussian config.

Our ρ slot layout (PDF Option C / model/executor/deform/rho_parser.py):
    0: E         Young's modulus
    1: ν         Poisson ratio
    2: ρ_m       mass density
    3..5: F[3]   external force vector (world frame)
    6: μ         friction coefficient
    7: damping
    8: dt

PhysGaussian config keys (subject to their JSON schema; verify against their
example configs after cloning the repo):
    material        "jelly" | "metal" | "sand" | "foam" | "plasticine"
    youngs_modulus  scalar
    poisson_ratio   scalar
    density         scalar
    external_force  [3] vector
    n_iterations    int
    substeps        int
    duration_secs   float
    fps             int

This module exposes one function: ``rho_to_physgaussian_config(rho, n_frames, fps)``.
Adjust if PhysGaussian's config schema in your version differs.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# Material classification heuristic
# ──────────────────────────────────────────────────────────────────────
# PartNet-Mobility objects are mostly rigid; PhysGaussian's closest preset
# is "metal".  We expose all five presets in case a particular trajectory's ρ
# clearly fits a softer category.

def _classify_material(E: float, nu: float, rho_m: float) -> str:
    """Pick the PhysGaussian material preset that best matches our ρ.

    Heuristic thresholds calibrated against PartNet-Mobility object types:
      hard rigid (cabinet/drawer/handle/door)  → metal      E > 1e10
      stiff plastic (lid)                       → plasticine 1e7 < E < 1e10
      soft (foam, sponge)                       → foam       1e4 < E < 1e7
      compliant (jelly, gel)                    → jelly      E < 1e4
    """
    if E >= 1e10:
        return "metal"
    if E >= 1e7:
        return "plasticine"
    if E >= 1e4:
        return "foam"
    return "jelly"


def rho_to_physgaussian_config(
    rho:        Sequence[float],         # length-9 vector
    n_frames:   int,
    fps:        int = 30,
    pbd_n_iter: int = 5,
    pbd_substeps: int = 2,
) -> Dict:
    """Convert our ρ → PhysGaussian config dict.

    The duration is ``n_frames / fps`` (in seconds); PhysGaussian internally
    decides its own substep count.  We pass our PBD's n_iter / n_substeps
    as hints when fields exist.
    """
    if len(rho) != 9:
        raise ValueError(f"ρ must have 9 dims (got {len(rho)})")

    E, nu, rho_m, fx, fy, fz, mu, damp, dt = [float(v) for v in rho]

    cfg = {
        "material":         _classify_material(E, nu, rho_m),
        "youngs_modulus":   max(E, 1e3),                      # PhysGaussian rejects E=0
        "poisson_ratio":    float(np.clip(nu, 0.0, 0.499)),   # avoid Poisson=0.5 (incompressible)
        "density":          max(rho_m, 1.0),                  # avoid 0
        "external_force":   [fx, fy, fz],
        "friction":         float(np.clip(mu, 0.0, 1.0)),
        "damping":          float(np.clip(damp, 0.0, 1.0)),
        "dt":               max(dt, 1e-4),                    # PhysGaussian wants positive dt
        "n_iterations":     pbd_n_iter,
        "substeps":         pbd_substeps,
        "duration_secs":    n_frames / fps,
        "fps":              fps,
    }
    return cfg


def default_rho_for_partnet_object(obj_category: str | None) -> Sequence[float]:
    """Fallback ρ for trajectories without a learned ρ (e.g. PhysGaussian
    on Dataset-B real video, where we have no GT physics).

    Categories grouped by typical material:
      cabinet / drawer / door / handle  → metal-like rigid
      microwave / oven / dishwasher     → metal-like rigid
      box / suitcase / lid              → plasticine-like
      everything else                   → metal default (safest for rigid sim)
    """
    rigid = {
        "cabinet", "drawer", "door", "handle", "microwave", "oven",
        "dishwasher", "refrigerator", "safe", "stapler",
    }
    cat = (obj_category or "").lower()
    if cat in rigid or any(k in cat for k in rigid):
        # metal-like
        return [2e11, 0.30, 7800.0, 0.0, 0.0, -9.81, 0.30, 0.05, 0.01]
    if any(k in cat for k in ("box", "suitcase", "lid", "kettle")):
        # plasticine-like
        return [1e8, 0.35, 800.0, 0.0, 0.0, -9.81, 0.40, 0.10, 0.01]
    # fallback: metal
    return [2e11, 0.30, 7800.0, 0.0, 0.0, -9.81, 0.30, 0.05, 0.01]
