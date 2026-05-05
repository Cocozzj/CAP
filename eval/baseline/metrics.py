"""Per-trajectory metric computation for baseline aggregation.

For each baseline that produced ``pred_4dgs.npz``, we compare predicted
Gaussian centers against ground-truth Gaussian centers reconstructed from
the trajectory.npz + init_gs.ply.

GT 4DGS construction
────────────────────
Our test data has:
  - init_gs.ply:         [N_full, 3] initial Gaussian centers (canonical)
  - trajectory.npz:      object_pose_world [T_gt, 7] (xyz + xyzw)

GT Gaussian centers at time t:
  mu_gt(t) = init_gs.mu @ R(pose_t)^T + translation(pose_t)

We then resample T_gt to T_pred frames so pred and GT align temporally.

Metrics implemented (geometric, no rendering):
  ade, fde, mpjpe          per-Gaussian Euclidean error
  energy_violation         total kinetic energy variance
  success_rate             FDE < threshold (default 5cm)
  multi_view_consistency   N/A here (requires rendering, see render_metrics.py)
  psnr, lpips, ssim        N/A here (requires rendering)
  closure_gap, inverse_gap N/A here (requires our Executor + tokens)
  diversity                N/A here (requires multi-sample inference)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .common import GS4DSequence, TrajMetrics


# ══════════════════════════════════════════════════════════════════════
# GT 4DGS reconstruction
# ══════════════════════════════════════════════════════════════════════

def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3×3 rotation matrix.  Vectorized over leading dims."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    R[..., 0, 0] = 1 - 2*(y*y + z*z); R[..., 0, 1] = 2*(x*y - z*w); R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] =     2*(x*y + z*w); R[..., 1, 1] = 1 - 2*(x*x + z*z); R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] =     2*(x*z - y*w); R[..., 2, 1] = 2*(y*z + x*w); R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R


def reconstruct_gt_centers(
    init_mu:           np.ndarray,        # [N, 3]
    object_pose_world: np.ndarray,        # [T_gt, 7]
    T_pred:            int,
) -> np.ndarray:
    """Apply GT object pose to init Gaussian centers → GT centers per frame.

    Subsamples T_gt frames evenly to T_pred to align with predicted sequence.
    Returns [T_pred, N, 3].
    """
    T_gt = int(object_pose_world.shape[0])
    if T_gt < 2:
        # Degenerate GT — broadcast init_mu
        return np.broadcast_to(init_mu[None], (T_pred,) + init_mu.shape).copy()

    # Reference pose at t=0 (so static objects produce mu_gt(0) = init_mu)
    pose0    = object_pose_world[0]
    R0       = _quat_xyzw_to_R(pose0[3:])
    t0       = pose0[:3]

    # Sample T_pred frames evenly from T_gt
    idx = np.linspace(0, T_gt - 1, T_pred, dtype=int)

    out = np.zeros((T_pred, init_mu.shape[0], 3), dtype=np.float32)
    for i, fi in enumerate(idx):
        Ri  = _quat_xyzw_to_R(object_pose_world[fi, 3:])
        ti  = object_pose_world[fi, :3]
        # Δ_pose = pose_i ∘ pose_0^-1
        dR  = Ri @ R0.T
        dt  = ti - dR @ t0
        out[i] = (init_mu @ dR.T) + dt[None]
    return out


def load_gt_centers(traj_dir: Path | str, init_mu: np.ndarray, T_pred: int) -> Optional[np.ndarray]:
    """Read trajectory.npz, reconstruct GT centers per frame.

    Returns None if trajectory.npz missing (Dataset-B has no GT).
    """
    p = Path(traj_dir) / "trajectory.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    if "object_pose_world" not in z.files:
        return None
    poses = z["object_pose_world"].astype(np.float32)
    return reconstruct_gt_centers(init_mu, poses, T_pred)


# ══════════════════════════════════════════════════════════════════════
# Geometric metrics
# ══════════════════════════════════════════════════════════════════════

def ade_fde_mpjpe(pred_mu: np.ndarray, gt_mu: np.ndarray) -> tuple[float, float, float]:
    """Compute trajectory error metrics on Gaussian centers.

    pred_mu, gt_mu: [T, N, 3]   (must match)

    Returns:
      ade   Average Displacement Error: mean over (T, N) of L2 distance
      fde   Final Displacement Error:   mean over N at t=T-1
      mpjpe Mean Per-Gaussian Position Error: same formula as ADE
            (we keep both for compatibility with motion-generation papers)
    """
    if pred_mu.shape != gt_mu.shape:
        # Different N — clip both to the smaller (4D-GS may have re-sampled)
        N = min(pred_mu.shape[1], gt_mu.shape[1])
        pred_mu = pred_mu[:, :N]
        gt_mu   = gt_mu[:, :N]
    err   = np.linalg.norm(pred_mu - gt_mu, axis=-1)        # [T, N]
    ade   = float(err.mean())
    fde   = float(err[-1].mean())
    mpjpe = ade   # same as ADE for our setup (no per-joint distinction)
    return ade, fde, mpjpe


def success_rate_position(pred_mu: np.ndarray, gt_mu: np.ndarray,
                           threshold_m: float = 0.05) -> int:
    """Binary success: 1 if FDE < threshold, else 0.

    Default threshold = 5 cm (matching 5cm translation step in our model).
    """
    if pred_mu.shape != gt_mu.shape:
        N = min(pred_mu.shape[1], gt_mu.shape[1])
        pred_mu = pred_mu[:, :N]
        gt_mu   = gt_mu[:, :N]
    fde = float(np.linalg.norm(pred_mu[-1] - gt_mu[-1], axis=-1).mean())
    return int(fde < threshold_m)


def energy_violation(pred_mu: np.ndarray, dt: float = 1.0 / 30.0) -> float:
    """Variance of total kinetic energy across timesteps (lower = more conserved).

    KE(t) = 0.5 * sum_n |v_n(t)|^2,  v_n(t) ≈ (mu(t+1) - mu(t)) / dt
    Return: std(KE) / mean(KE)   (coefficient of variation, scale-free)
    """
    T = int(pred_mu.shape[0])
    if T < 2:
        return 0.0
    v = (pred_mu[1:] - pred_mu[:-1]) / max(dt, 1e-6)         # [T-1, N, 3]
    ke = 0.5 * (v ** 2).sum(axis=(1, 2))                     # [T-1]
    if ke.mean() < 1e-12:
        return 0.0
    return float(ke.std() / (ke.mean() + 1e-12))


def smoothness(pred_mu: np.ndarray) -> float:
    """L2 norm of acceleration (lower = smoother).  Useful sanity metric."""
    if pred_mu.shape[0] < 3:
        return 0.0
    a = pred_mu[2:] - 2 * pred_mu[1:-1] + pred_mu[:-2]       # [T-2, N, 3]
    return float(np.linalg.norm(a, axis=-1).mean())


# ══════════════════════════════════════════════════════════════════════
# All-in-one per-trajectory metric
# ══════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    pred_seq:    GS4DSequence,
    gt_centers:  Optional[np.ndarray],          # [T, N, 3] or None (missing GT)
    fps:         float = 30.0,
) -> TrajMetrics:
    """Compute every geometric metric we can given pred + GT.

    Visual / algebraic / diversity metrics are NOT computed here — they need
    rendering / our model / multi-sample inference.  They stay None.
    """
    pred_mu = pred_seq.mu                                     # [T, N, 3]

    if gt_centers is None:
        # Dataset-B or missing GT — only physics-style metrics are available
        return TrajMetrics(
            energy_violation=energy_violation(pred_mu, dt=1.0/fps),
            notes="no_gt_trajectory",
        )

    # Align T (caller should pre-resample, but safe fallback here)
    if pred_mu.shape[0] != gt_centers.shape[0]:
        T_min = min(pred_mu.shape[0], gt_centers.shape[0])
        pred_mu_aligned = pred_mu[:T_min]
        gt_aligned      = gt_centers[:T_min]
    else:
        pred_mu_aligned = pred_mu
        gt_aligned      = gt_centers

    ade, fde, mpjpe = ade_fde_mpjpe(pred_mu_aligned, gt_aligned)
    succ = success_rate_position(pred_mu_aligned, gt_aligned)

    return TrajMetrics(
        ade=ade, fde=fde, mpjpe=mpjpe,
        success=succ,
        energy_violation=energy_violation(pred_mu, dt=1.0/fps),
        notes="ok",
    )
