"""Parse PhysDreamer's per-frame output → our GS4DSequence.

PhysDreamer writes its outputs in one of these schemas (we try each):

  Schema A (per-frame PLYs):
      <out_dir>/frames/frame_000.ply, frame_001.ply, ...

  Schema B (single npz):
      <out_dir>/sim_output.npz       with keys: mu, cov, sh, opacity, scale

  Schema C (compact npz, deformation-only):
      <out_dir>/deformations.npz     with keys: delta_mu, delta_cov  (per-frame)
      → applied to canonical Gaussians from input init_gs.ply

After cloning PhysDreamer, verify which schema applies and adjust here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from ..common import GS4DSequence
from ..kinematics import quat_log_scale_to_full_cov


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _load_ply_centers_only(ply_path: Path) -> Optional[np.ndarray]:
    """Read PLY → [N, 3] Gaussian centers (mu)."""
    try:
        from plyfile import PlyData
    except ImportError:
        return None
    if not ply_path.exists():
        return None
    v = PlyData.read(str(ply_path))["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)


def _load_ply_full(ply_path: Path, c_sh: int = 48) -> Optional[dict]:
    """Read PLY → {mu, cov_quat, log_scale, sh, opacity}, all numpy arrays."""
    try:
        from plyfile import PlyData
    except ImportError:
        return None
    if not ply_path.exists():
        return None
    v = PlyData.read(str(ply_path))["vertex"].data
    N = len(v)

    def _stack(prefix: str, dim: int) -> np.ndarray:
        try:
            return np.stack([v[f"{prefix}_{i}"] for i in range(dim)], axis=-1).astype(np.float32)
        except (KeyError, ValueError):
            return np.zeros((N, dim), dtype=np.float32)

    mu       = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    cov_quat = _stack("rot", 4)             # [N, 4] quaternion
    log_sc   = _stack("scale", 3)
    sh_dc    = _stack("f_dc", 3)            # [N, 3] DC term only
    opac_arr = (v["opacity"][..., None].astype(np.float32)
                if "opacity" in v.dtype.names else np.zeros((N, 1), dtype=np.float32))

    # Pad sh to c_sh dim if only DC exists
    if sh_dc.shape[-1] < c_sh:
        pad = np.zeros((N, c_sh - sh_dc.shape[-1]), dtype=np.float32)
        sh = np.concatenate([sh_dc, pad], axis=-1)
    else:
        sh = sh_dc[..., :c_sh]
    return {"mu": mu, "cov_quat": cov_quat, "log_scale": log_sc, "sh": sh, "opacity": opac_arr}


# ──────────────────────────────────────────────────────────────────────
# Schema parsers
# ──────────────────────────────────────────────────────────────────────

def _parse_schema_A_per_frame_ply(out_dir: Path) -> Optional[GS4DSequence]:
    """Per-frame PLY files in <out_dir>/frames/."""
    plys = sorted((out_dir / "frames").glob("frame_*.ply"))
    if not plys:
        plys = sorted(out_dir.glob("frame_*.ply"))
    if not plys:
        return None

    first = _load_ply_full(plys[0])
    if first is None:
        return None
    N = first["mu"].shape[0]
    T = len(plys)

    mu_t  = np.zeros((T, N, 3),    dtype=np.float32)
    cov_t = np.zeros((T, N, 3, 3), dtype=np.float32)
    for t, p in enumerate(plys):
        d = _load_ply_full(p)
        if d is None or d["mu"].shape[0] != N:
            continue
        mu_t[t]  = d["mu"]
        cov_t[t] = quat_log_scale_to_full_cov(d["cov_quat"], d["log_scale"])

    sh_full      = first["sh"]
    opacity_full = first["opacity"]
    scale_full   = first["log_scale"]

    return GS4DSequence(
        mu=mu_t, cov=cov_t,
        sh=np.broadcast_to(sh_full[None],      (T,) + sh_full.shape).copy(),
        opacity=np.broadcast_to(opacity_full[None], (T,) + opacity_full.shape).copy(),
        scale=np.broadcast_to(scale_full[None],   (T,) + scale_full.shape).copy(),
    )


def _parse_schema_B_single_npz(out_dir: Path) -> Optional[GS4DSequence]:
    """Single .npz with full mu/cov/sh/... arrays."""
    candidates = (
        out_dir / "sim_output.npz",
        out_dir / "output.npz",
        out_dir / "physdreamer_output.npz",
    )
    for cand in candidates:
        if cand.exists():
            try:
                z = np.load(cand)
                return GS4DSequence(
                    mu=z["mu"], cov=z["cov"], sh=z["sh"],
                    opacity=z["opacity"], scale=z["scale"],
                )
            except KeyError:
                continue
    return None


def _parse_schema_C_deformation_only(out_dir: Path, init_ply: Path) -> Optional[GS4DSequence]:
    """Deformations.npz (delta_mu / delta_cov) applied to init_gs.ply."""
    delta_path = out_dir / "deformations.npz"
    if not delta_path.exists():
        return None
    try:
        z = np.load(delta_path)
    except Exception:
        return None
    if "delta_mu" not in z.files:
        return None

    init = _load_ply_full(init_ply)
    if init is None:
        return None
    N = init["mu"].shape[0]
    delta_mu = z["delta_mu"]                                  # [T, N, 3]
    if delta_mu.ndim != 3 or delta_mu.shape[1] != N:
        return None
    T = delta_mu.shape[0]

    mu_t = init["mu"][None] + delta_mu                        # [T, N, 3]
    if "delta_cov" in z.files and z["delta_cov"].shape == (T, N, 3, 3):
        cov0  = quat_log_scale_to_full_cov(init["cov_quat"], init["log_scale"])
        cov_t = cov0[None] + z["delta_cov"]
    else:
        # Broadcast initial cov across all frames (no rotation)
        cov0  = quat_log_scale_to_full_cov(init["cov_quat"], init["log_scale"])
        cov_t = np.broadcast_to(cov0[None], (T,) + cov0.shape).copy()

    sh_full      = init["sh"]
    opacity_full = init["opacity"]
    scale_full   = init["log_scale"]
    return GS4DSequence(
        mu=mu_t.astype(np.float32),
        cov=cov_t.astype(np.float32),
        sh=np.broadcast_to(sh_full[None],      (T,) + sh_full.shape).copy(),
        opacity=np.broadcast_to(opacity_full[None], (T,) + opacity_full.shape).copy(),
        scale=np.broadcast_to(scale_full[None],   (T,) + scale_full.shape).copy(),
    )


# ──────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────

def parse_physdreamer_output(out_dir: Path | str,
                              init_ply: Optional[Path | str] = None
                              ) -> Optional[GS4DSequence]:
    """Try each known PhysDreamer output schema; return first match."""
    out_dir = Path(out_dir)
    init_ply = Path(init_ply) if init_ply is not None else (out_dir.parent / "init_gs.ply")

    seq = _parse_schema_A_per_frame_ply(out_dir)
    if seq is not None:
        return seq

    seq = _parse_schema_B_single_npz(out_dir)
    if seq is not None:
        return seq

    seq = _parse_schema_C_deformation_only(out_dir, init_ply)
    if seq is not None:
        return seq

    return None
