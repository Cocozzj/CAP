"""
rigid/rotation_group.py — Discrete rotation group  H ⊂ SO(3)  (chiral octahedral, |H| = 24).

Provides the cube24 rotation set used by:
  - Encoder.ActionTokenizer  (codebook initialisation — see encoder/ActionTokenizer.py)
  - Executor cross-object transfer (mapping actions across canonical frames)
  - Loss functions that need a discrete rotation reference

Note: With the new design, the rigid execution path no longer goes through
``DiscreteRotationGroup`` — Encoder.head_h already outputs a continuous SO(3)
matrix via 6D + Gram-Schmidt.  This module is kept available for utilities
that still need cube24 lookup (e.g. inverse element queries).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════════
# §1  Single axis-angle → rotation matrix (Rodrigues)
# ═══════════════════════════════════════════════════════════════════════════

def _axis_angle_to_matrix(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """Rodrigues' formula for a single axis-angle pair.

    Args:
        axis:  [3]   rotation axis (need not be unit-length)
        angle: float scalar angle in radians

    Returns:
        R: [3, 3]    rotation matrix
    """
    a = axis / axis.norm()
    K = torch.tensor([
        [0.,    -a[2],  a[1]],
        [a[2],  0.,    -a[0]],
        [-a[1], a[0],  0.  ],
    ], dtype=a.dtype)
    I = torch.eye(3, dtype=a.dtype)
    return I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


# ═══════════════════════════════════════════════════════════════════════════
# §2  Build the 24 chiral octahedral rotations
# ═══════════════════════════════════════════════════════════════════════════

def build_discrete_rotation_group(
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Enumerate the 24 chiral (proper) octahedral rotations.

    Generation:
      - 1 identity
      - 9 face-axis rotations: 90°, 180°, 270° around x, y, z axes
      - 8 body-diagonal rotations: ±120° around 4 cube diagonals
      - 6 edge-midpoint rotations: 180° around 6 edge axes

    Returns:
        H: [24, 3, 3]  rotation matrices, all with det = +1
    """
    rots: list[torch.Tensor] = []

    # Identity
    rots.append(torch.eye(3))

    # 90°, 180°, 270° around coordinate axes  → 9
    for ax in [torch.tensor([1., 0., 0.]),
               torch.tensor([0., 1., 0.]),
               torch.tensor([0., 0., 1.])]:
        for k in (1, 2, 3):
            rots.append(_axis_angle_to_matrix(ax, k * math.pi / 2))

    # 120°, 240° around body diagonals  → 8
    for d in [torch.tensor([1., 1., 1.]),  torch.tensor([1., 1., -1.]),
              torch.tensor([1., -1., 1.]), torch.tensor([-1., 1., 1.])]:
        for k in (1, 2):
            rots.append(_axis_angle_to_matrix(d, k * 2 * math.pi / 3))

    # 180° around edge midpoints  → 6
    for e in [torch.tensor([1., 1., 0.]), torch.tensor([1., -1., 0.]),
              torch.tensor([1., 0., 1.]), torch.tensor([1., 0., -1.]),
              torch.tensor([0., 1., 1.]), torch.tensor([0., 1., -1.])]:
        rots.append(_axis_angle_to_matrix(e, math.pi))

    assert len(rots) == 24, f"Expected 24 rotations, got {len(rots)}"
    return torch.stack(rots, dim=0).to(device)


# ═══════════════════════════════════════════════════════════════════════════
# §3  DiscreteRotationGroup module
# ═══════════════════════════════════════════════════════════════════════════

class DiscreteRotationGroup(nn.Module):
    """Non-learnable buffer holding cube24 (H) and providing index ↔ matrix
    lookups + an inverse table.

    Use cases:
      - Look up R_h given an integer index in [0, 24)
      - Find the inverse element index for a given rotation
      - Reference H for cross-object transfer or loss computation

    With the new Encoder design (head_h outputs continuous SO(3) directly),
    this group is NOT used in the rigid forward path.  Kept as a utility.
    """

    def __init__(self) -> None:
        super().__init__()
        H = build_discrete_rotation_group()                                # [24, 3, 3]
        self.register_buffer("H", H)

        # Pre-compute inverse lookup: inv_table[i] = j  s.t.  H[j] ≈ H[i]^T
        with torch.no_grad():
            H_inv = H.transpose(-2, -1)                                    # [24, 3, 3]
            # Closest H element to each H_inv by Frobenius distance
            diff = (H_inv.unsqueeze(1) - H.unsqueeze(0)).norm(dim=(-2, -1))  # [24, 24]
            inv_table = diff.argmin(dim=1)                                 # [24]
        self.register_buffer("inv_table", inv_table)

    @property
    def size(self) -> int:
        return int(self.H.shape[0])

    def index_to_matrix(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [*batch] long  →  R_h: [*batch, 3, 3]."""
        return self.H[idx]

    def inverse_index(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: [*batch] long  →  inv_idx: [*batch] long  s.t.  H[inv_idx] = H[idx]^T."""
        return self.inv_table[idx]
