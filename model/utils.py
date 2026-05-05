from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict


# ═══════════════════════════════════════════════════════════════════════════
# §1  Data structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CanonicalFrame:
    """
    Per-object canonical frame  Φ_o = (R, t)  mapping world → canonical.
        x_can = (x_world - t) @ R            (row-vector convention)
        x_world = x_can @ R^T + t
    """
    R_w2c: torch.Tensor     # [B, K, 3, 3]  world-to-canonical rotation
    t_w2c: torch.Tensor     # [B, K, 3]     world-to-canonical translation (= COM)

    @property
    def R_c2w(self) -> torch.Tensor:
        return self.R_w2c.transpose(-2, -1)

    @property
    def t_c2w(self) -> torch.Tensor:
        return self.t_w2c  # same point — just used with R_c2w


@dataclass
class SceneState:
    """Padded 3DGS scene (the only scene representation used by Executor).

    Layout: B batches × K object slots × N_max Gaussians.
    Real Gaussians have ``mask = True``; padded slots have ``opacity = 0`` and
    ``mask = False``.  All aggregation operations MUST go through the
    ``masked_*`` helpers below to avoid bias from padded zeros.

    Fields
    ------
    mu          [B, K, N_max, 3]        Gaussian centres (world space)
    cov         [B, K, N_max, 3, 3]     Covariance matrices (world space, SPD)
    sh          [B, K, N_max, C_sh]     SH colour coefficients
    opacity     [B, K, N_max, 1]        Opacity ∈ [0, 1]; 0 for padded
    scale       [B, K, N_max, 3]        Log-scale per axis
    phi         CanonicalFrame           Per-slot canonical frames Φ_o
                                         R_w2c [B, K, 3, 3], t_w2c [B, K, 3]
    mask        [B, K, N_max] bool      True = real Gaussian, False = padding
    R_obj_world [B, K, 3, 3]            Accumulated world-space rotation
                                         (identity at t=0; updated per token)
    """
    mu:          torch.Tensor     # [B, K, N_max, 3]
    cov:         torch.Tensor     # [B, K, N_max, 3, 3]
    sh:          torch.Tensor     # [B, K, N_max, C_sh]
    opacity:     torch.Tensor     # [B, K, N_max, 1]
    scale:       torch.Tensor     # [B, K, N_max, 3]
    phi:         CanonicalFrame
    mask:        torch.Tensor     # [B, K, N_max] bool
    R_obj_world: Optional[torch.Tensor] = None   # [B, K, 3, 3]

    def __post_init__(self):
        if self.R_obj_world is None:
            B, K = self.mu.shape[:2]
            self.R_obj_world = torch.eye(
                3, device=self.mu.device, dtype=self.mu.dtype
            ).view(1, 1, 3, 3).expand(B, K, -1, -1).clone()

    # ── shape accessors ───────────────────────────────────────────────
    @property
    def B(self) -> int:
        return int(self.mu.shape[0])

    @property
    def K(self) -> int:
        return int(self.mu.shape[1])

    @property
    def N_max(self) -> int:
        return int(self.mu.shape[2])

    @property
    def padding_waste(self) -> float:
        """Fraction of [B, K, N_max] positions that are padding.

        0.0 = perfectly packed (no waste)
        1.0 = all positions are padding (degenerate)

        Use this as a sanity check during training:
            if state.padding_waste > 0.7:
                warnings.warn("Padding waste is high; consider bucketed sampler.")
        """
        n_real = float(self.mask.sum().item())
        n_total = float(self.mask.numel())
        return 1.0 - n_real / max(n_total, 1.0)

    def clone(self) -> "SceneState":
        return SceneState(
            mu          = self.mu.clone(),
            cov         = self.cov.clone(),
            sh          = self.sh.clone(),
            opacity     = self.opacity.clone(),
            scale       = self.scale.clone(),
            phi         = CanonicalFrame(
                R_w2c=self.phi.R_w2c.clone(),
                t_w2c=self.phi.t_w2c.clone(),
            ),
            mask        = self.mask.clone(),
            R_obj_world = self.R_obj_world.clone(),
        )


@dataclass
class GSParameter:
    """Single 3DGS scene for ONE sample: N Gaussian primitives (no batch dim).

    This is the format produced by 3DGS preprocessing pipelines
    (e.g. .ply / .splat files).  Use ``quaternion_to_full_cov`` to convert
    the (quat, log_scale) parameterisation into a full 3×3 SPD covariance.
    """
    mu: torch.Tensor          # [N, 3]     Gaussian centres (world)
    cov: torch.Tensor         # [N, 4]     covariance parameterisation (quaternion)
    scale: torch.Tensor       # [N, 3]     log-scale vector
    sh: torch.Tensor          # [N, C_sh]  SH coefficients
    opacity: torch.Tensor     # [N, 1]     opacity  ∈ [0, 1]

    @property
    def num_gaussians(self) -> int:
        return int(self.mu.shape[-2])

    def __len__(self) -> int:
        """Number of Gaussians (== mu.shape[0])."""
        return int(self.mu.shape[0])

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "GSParameter":
        """Move all tensors to a device / dtype.

        ``non_blocking=True`` enables async copy when used with pinned memory
        (set ``pin_memory=True`` in DataLoader).
        """
        kw = {"device": device, "non_blocking": non_blocking}
        if dtype is not None:
            kw["dtype"] = dtype
        return GSParameter(
            mu=self.mu.to(**kw),
            cov=self.cov.to(**kw),
            scale=self.scale.to(**kw),
            sh=self.sh.to(**kw),
            opacity=self.opacity.to(**kw),
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Dict view for SlotAttention / SlotGSBinder compatibility."""
        return {
            "mu": self.mu, "cov": self.cov, "scale": self.scale,
            "sh": self.sh, "opacity": self.opacity,
        }


# ═══════════════════════════════════════════════════════════════════════════
# §2  Mask-aware reductions
#     Every aggregation over the padded N dimension MUST use these helpers.
#     Vanilla .mean() / .sum() over padded SceneState → biased toward origin.
# ═══════════════════════════════════════════════════════════════════════════

def _broadcast_mask(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """Expand mask trailing dims to match a target tensor (e.g. [B, K, N] → [B, K, N, 1])."""
    m = mask.float()
    while m.dim() < target_ndim:
        m = m.unsqueeze(-1)
    return m


def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor,
    dim: int,
    keepdim: bool = False,
) -> torch.Tensor:
    """Mean of x over `dim`, ignoring mask=False positions.

    Mathematically equivalent to taking the mean over only the valid entries:
        masked_mean(x_padded, mask, dim) ≡ x_real.mean(dim)

    Returns 0 where every position along `dim` is masked out (no NaN).
    """
    m = _broadcast_mask(mask, x.dim())
    n_valid = mask.float().sum(dim=dim, keepdim=keepdim).clamp(min=1)
    # n_valid currently lives in mask's dim space; expand to x's dim space
    while n_valid.dim() < (x.dim() if keepdim else x.dim() - 1):
        n_valid = n_valid.unsqueeze(-1)
    # NaN-SAFE: ``x * m`` would propagate NaN from masked-out positions
    # (IEEE 754: NaN * 0 = NaN), poisoning the mean.  Use torch.where to
    # actually replace masked-out positions with 0 before summing.
    x_masked = torch.where(m.bool(), x, torch.zeros_like(x))
    return x_masked.sum(dim=dim, keepdim=keepdim) / n_valid


# ═══════════════════════════════════════════════════════════════════════════
# §3  Scene construction
# ═══════════════════════════════════════════════════════════════════════════

def quaternion_to_full_cov(
    quat: torch.Tensor,         # [..., 4]  (w, x, y, z) quaternion
    log_scale: torch.Tensor,    # [..., 3]  log-scale per axis
    scale_floor: float = 1e-6,
) -> torch.Tensor:
    """
    Convert (quaternion, log-scale) parameterisation to full 3×3 SPD covariance:

        Σ = R · diag(s²) · R^T   = (R · diag(s)) · (R · diag(s))^T

    where R = quaternion_to_matrix(quat), s = exp(log_scale) (clamped).

    Args:
        quat:        [..., 4]  unit quaternions (w, x, y, z); will be normalised
        log_scale:   [..., 3]  per-axis log-scale
        scale_floor: lower bound on s = exp(log_scale) for numerical stability

    Returns:
        cov:         [..., 3, 3] symmetric positive-definite covariance matrix
    """
    R = quaternion_to_matrix(quat)                          # [..., 3, 3]
    s = log_scale.exp().clamp(min=scale_floor)              # [..., 3]
    # R · diag(s): scale each column of R by s (broadcast multiply)
    RS = R * s.unsqueeze(-2)                                # [..., 3, 3]
    return RS @ RS.transpose(-2, -1)                        # SPD


def build_scene_state(
    gs_params: List[GSParameter],            # B samples (Dataset)
    phi: CanonicalFrame,                     # [B, K, ...] (Encoder)
    assignment: List[torch.Tensor],          # B × [N_b, K] one-hot (Encoder)
    R_obj_world: Optional[torch.Tensor] = None,  # [B, K, 3, 3] optional
) -> SceneState:
    """Build a padded SceneState from Dataset + Encoder outputs.

    Combines:
      - Dataset:  raw per-sample GSParameter (no batch dim)
      - Encoder:  phi (canonical frames) + assignment (slot binding)

    into one padded 4D SceneState [B, K, N_max, ...] suitable for the Executor.
    Padded positions get ``opacity = 0`` and ``mask = False``.

    Args:
        gs_params:    List of B GSParameter (each with N_b Gaussians)
        phi:          CanonicalFrame with R_w2c [B, K, 3, 3], t_w2c [B, K, 3]
        assignment:   List of B Tensors [N_b, K] one-hot (or hard) slot assignment
        R_obj_world:  optional [B, K, 3, 3]; defaults to identity

    Returns:
        SceneState with all per-sample data padded to N_max = max object size in batch.
    """
    B = len(gs_params)
    K = int(phi.R_w2c.shape[-3])
    device = gs_params[0].mu.device
    dtype = gs_params[0].mu.dtype
    C_sh = int(gs_params[0].sh.shape[-1])

    # ── Determine per-(b, k) sizes and N_max ──────────────────────────
    # owners[b]: [N_b] long, slot index in [0, K)
    owners = [a.argmax(dim=-1) for a in assignment]
    sizes: List[List[int]] = []
    for b in range(B):
        per_slot_sizes = [int((owners[b] == k).sum().item()) for k in range(K)]
        sizes.append(per_slot_sizes)
    N_max = max((max(row) for row in sizes), default=0)
    N_max = max(N_max, 1)  # at least 1 to keep tensor shapes well-defined

    # ── Allocate padded tensors (all zeros + mask=False) ──────────────
    mu_pad      = torch.zeros(B, K, N_max, 3,    device=device, dtype=dtype)
    cov_pad     = torch.zeros(B, K, N_max, 3, 3, device=device, dtype=dtype)
    sh_pad      = torch.zeros(B, K, N_max, C_sh, device=device, dtype=dtype)
    opacity_pad = torch.zeros(B, K, N_max, 1,    device=device, dtype=dtype)
    scale_pad   = torch.zeros(B, K, N_max, 3,    device=device, dtype=dtype)
    mask        = torch.zeros(B, K, N_max,        device=device, dtype=torch.bool)

    # ── Fill: scatter each (b, k) slice + convert quaternion cov to SPD ──
    for b in range(B):
        g = gs_params[b]
        # Convert quaternion + log_scale → full 3×3 SPD covariance once per sample
        cov_full = quaternion_to_full_cov(g.cov, g.scale)        # [N_b, 3, 3]
        for k in range(K):
            n_k = sizes[b][k]
            if n_k == 0:
                continue
            idx = (owners[b] == k).nonzero(as_tuple=True)[0]      # [n_k]
            mu_pad[b, k, :n_k]      = g.mu[idx]
            cov_pad[b, k, :n_k]     = cov_full[idx]
            sh_pad[b, k, :n_k]      = g.sh[idx]
            opacity_pad[b, k, :n_k] = g.opacity[idx]
            scale_pad[b, k, :n_k]   = g.scale[idx]
            mask[b, k, :n_k]        = True

    return SceneState(
        mu=mu_pad,
        cov=cov_pad,
        sh=sh_pad,
        opacity=opacity_pad,
        scale=scale_pad,
        phi=phi,
        mask=mask,
        R_obj_world=R_obj_world,   # None → auto-init to identity in __post_init__
    )


# ═══════════════════════════════════════════════════════════════════════════
# §4  SPD covariance utilities  (linear ↔ log-Euclidean, SPD projection)
#
# Used by:
#   - executor/deform/sim.py           (covariance updates after deformation)
#   - executor/executor.py             (defensive SPD projection at world out)
#   - quaternion_to_full_cov + losses
# ═══════════════════════════════════════════════════════════════════════════

def _safe_eigh(M: torch.Tensor):
    """Symmetric eigendecomposition with NaN guard, AMP cast, and CUDA fallback.

    PyTorch's ``torch.linalg.eigh`` can fail on near-singular matrices under
    AMP (fp16/bf16) or on CUDA.  We:
      1. Cast to float32 if needed
      2. Replace NaN/inf with finite values
      3. Symmetrize to defend against round-off asymmetry
      4. Fall back to CPU eigh if CUDA decomposition errors out
    """
    orig_dtype = M.dtype
    if M.dtype in (torch.float16, torch.bfloat16):
        M = M.float()
    M = torch.nan_to_num(M, nan=0.0, posinf=1e4, neginf=-1e4)
    M = 0.5 * (M + M.transpose(-2, -1))
    try:
        eigvals, eigvecs = torch.linalg.eigh(M)
    except torch._C._LinAlgError:
        # CUDA eigh occasionally fails — retry on CPU then move back
        eigvals, eigvecs = torch.linalg.eigh(M.cpu())
        eigvals = eigvals.to(M.device)
        eigvecs = eigvecs.to(M.device)
    return eigvals.to(orig_dtype), eigvecs.to(orig_dtype)


def cov_to_log_euclidean(Sigma: torch.Tensor) -> torch.Tensor:
    """SPD covariance Σ → log(Σ) (symmetric, no positive constraint).

    The log-Euclidean parameterisation lets us add covariance increments
    additively (linear in log space) while still recovering an SPD result via
    matrix exponential.  Use ``log_euclidean_to_cov`` for the inverse map.
    """
    eigvals, eigvecs = _safe_eigh(Sigma)
    log_eigvals = eigvals.clamp(min=1e-8).log()
    return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-2, -1)


def log_euclidean_to_cov(logS: torch.Tensor) -> torch.Tensor:
    """log(Σ) → Σ (SPD)."""
    eigvals, eigvecs = _safe_eigh(logS)
    return eigvecs @ torch.diag_embed(eigvals.exp()) @ eigvecs.transpose(-2, -1)


def project_spd(Sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Project an arbitrary symmetric matrix to the nearest SPD by clamping
    eigenvalues to at least ``eps``.  Used after non-trivial covariance
    updates to guarantee Σ remains positive definite."""
    eigvals, eigvecs = _safe_eigh(Sigma)
    eigvals = eigvals.clamp(min=eps)
    return eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)


# ═══════════════════════════════════════════════════════════════════════════
# §5  Core SE(3) / so(3) operations
# ═══════════════════════════════════════════════════════════════════════════

def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """
    Exponential map  so(3) → SO(3)  via Rodrigues' formula.

    Args:
        omega: [..., 3]  axis-angle vectors (ξ in the paper)

    Returns:
        R: [..., 3, 3]  rotation matrices
    """
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=1e-8)        # [..., 1]
    axis = omega / theta                                              # [..., 3]

    # skew-symmetric matrix  [axis]_×
    K = skew_symmetric(axis)                                          # [..., 3, 3]

    theta = theta.unsqueeze(-1)                                       # [..., 1, 1]
    I = torch.eye(3, device=omega.device, dtype=omega.dtype)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Logarithmic map  SO(3) → so(3).

    Args:
        R: [..., 3, 3]  rotation matrices

    Returns:
        omega: [..., 3]  axis-angle vectors
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    theta = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))  # [...,]

    # Extract axis from skew part of (R - Rᵀ) / (2 sin θ)
    skew = (R - R.transpose(-1, -2)) / (2 * theta.clamp(min=1e-8).unsqueeze(-1).unsqueeze(-1))
    omega = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    omega = omega * theta.unsqueeze(-1)
    return omega


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """
    Build skew-symmetric matrix from 3-vector.

    Args:
        v: [..., 3]

    Returns:
        K: [..., 3, 3]  where K × u = v × u
    """
    zeros = torch.zeros_like(v[..., 0])
    K = torch.stack([
        zeros,    -v[..., 2],  v[..., 1],
        v[..., 2],  zeros,    -v[..., 0],
       -v[..., 1],  v[..., 0],  zeros
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)
    return K


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert unit quaternion (w, x, y, z) → rotation matrix.

    Args:
        q: [..., 4]  unit quaternions

    Returns:
        R: [..., 3, 3]
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1).reshape(*q.shape[:-1], 3, 3)
    return R


# ═══════════════════════════════════════════════════════════════════════════
# §6  Canonical frame COMPUTATION
#     Used by Encoder Stage 1.4 (per-object).
# ═══════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def compute_canonical_frame(
    positions: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    min_points: int = 3,
    eps: float = 1e-6,
) -> CanonicalFrame:
    """
    Compute canonical frame Φ_o = (R_o, t_o) for a set of Gaussian centers
    via center-of-mass translation + PCA principal axes alignment.

    Supports both batched [B, N, 3] and unbatched [N, 3] inputs.

    Includes robustness features:
      - min_points guard (returns identity frame for degenerate objects)
      - covariance symmetrization + epsilon stabilization
      - sign resolution (consistent axis orientation across runs)
      - QR re-orthonormalization
      - right-handedness enforcement (det R = +1)

    Args:
        positions: [B, N, 3] or [N, 3]  Gaussian centers belonging to object o
        weights:   [B, N] or [N]         optional per-Gaussian weights (e.g. opacity)
                                          If None, uniform weighting is used.
        min_points: int                   minimum points for valid PCA (else identity)
        eps:        float                 numerical stabilizer

    Returns:
        CanonicalFrame with:
            R: [B, 3, 3] or [3, 3]  rotation (world → canonical)
            t: [B, 3] or [3]        center of mass
    """
    unbatched = positions.dim() == 2
    if unbatched:
        positions = positions.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)

    B, N, _ = positions.shape
    device, dtype = positions.device, positions.dtype

    all_R = []
    all_t = []

    for b in range(B):
        mu = positions[b]                                     # [N, 3]
        w = weights[b] if weights is not None else None       # [N] or None
        n = mu.shape[0]

        # ── Edge case: too few points → identity frame ──
        if n < min_points:
            t = mu.mean(dim=0) if n > 0 else torch.zeros(3, device=device, dtype=dtype)
            R = torch.eye(3, device=device, dtype=dtype)
            all_R.append(R)
            all_t.append(t)
            continue

        # ── Weighted center of mass ──
        if w is None:
            t = mu.mean(dim=0)
            centered = mu - t
            cov = (centered.T @ centered) / n
        else:
            w = w.clamp_min(0) + eps
            wsum = w.sum().clamp_min(eps)
            t = (mu * w.unsqueeze(-1)).sum(0) / wsum
            centered = mu - t
            cov = (centered.T @ (centered * w.unsqueeze(-1))) / wsum

        # ── Stabilize covariance ──
        cov = (cov + cov.T) * 0.5
        cov = cov + eps * torch.eye(3, device=device, dtype=dtype)

        # ── PCA via eigendecomposition ──
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvectors = eigenvectors[:, order]

        # ── Sign resolution FIRST: consistent axis orientation ──
        projections = centered @ eigenvectors               # [N, 3]
        signs = projections.mean(dim=0).sign()
        signs[signs == 0] = 1.0
        eigenvectors = eigenvectors * signs.unsqueeze(0)

        # ── QR re-orthonormalize (preserves sign choices) ──
        eigenvectors, _ = torch.linalg.qr(eigenvectors)

        # ── Right-handedness ──
        R = eigenvectors.T
        if torch.linalg.det(R) < 0:
            R[2] = -R[2]

        all_R.append(R)
        all_t.append(t)

    R_out = torch.stack(all_R, dim=0)                         # [B, 3, 3]
    t_out = torch.stack(all_t, dim=0)                         # [B, 3]

    if unbatched:
        R_out = R_out.squeeze(0)
        t_out = t_out.squeeze(0)

    return CanonicalFrame(R_w2c=R_out, t_w2c=t_out)


# ═══════════════════════════════════════════════════════════════════════════
# §7  Canonical frame APPLICATION (used by Executor)
# ═══════════════════════════════════════════════════════════════════════════

def to_canonical(
    mu: torch.Tensor,
    cov: torch.Tensor,
    phi: CanonicalFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map Gaussians from world space → canonical space.

    Row-vector position:  μ_can = (μ - t) @ R_w2c
    Covariance:           Σ_can = R_w2c^T Σ R_w2c

    Supports arbitrary leading dimensions:
        [B, N, 3]        with R_w2c [B, 3, 3]
        [B, K, N, 3]     with R_w2c [B, K, 3, 3]

    Args:
        mu:   [..., N, 3]      Gaussian centers (world)
        cov:  [..., N, 3, 3]   Gaussian covariance matrices (world)
        phi:  CanonicalFrame   (R_w2c: [..., 3, 3], t_w2c: [..., 3])

    Returns:
        mu_can:  [..., N, 3]
        cov_can: [..., N, 3, 3]
    """
    R = phi.R_w2c.unsqueeze(-3)                           # [..., 1, 3, 3]
    t = phi.t_w2c.unsqueeze(-2)                           # [..., 1, 3]
    mu_c  = (mu - t) @ R.squeeze(-3)                      # [..., N, 3]
    cov_c = R.transpose(-2, -1) @ cov @ R                 # [..., N, 3, 3]
    return mu_c, cov_c


def from_canonical(
    mu_c: torch.Tensor,
    cov_c: torch.Tensor,
    phi: CanonicalFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Map Gaussians from canonical space → world space.

    Row-vector position:  μ_world = μ_can @ R_c2w + t
    Covariance:           Σ_world = R_w2c Σ_can R_w2c^T

    Supports arbitrary leading dimensions:
        [B, N, 3]        with R_w2c [B, 3, 3]
        [B, K, N, 3]     with R_w2c [B, K, 3, 3]

    Args:
        mu_c:  [..., N, 3]      Gaussian centers (canonical)
        cov_c: [..., N, 3, 3]   Gaussian covariance matrices (canonical)
        phi:   CanonicalFrame   (R_w2c: [..., 3, 3], t_w2c: [..., 3])

    Returns:
        mu_w:  [..., N, 3]
        cov_w: [..., N, 3, 3]
    """
    Rinv = phi.R_c2w.unsqueeze(-3)                        # [..., 1, 3, 3]  = R_w2c^T
    t    = phi.t_w2c.unsqueeze(-2)                        # [..., 1, 3]
    mu_w  = mu_c @ Rinv.squeeze(-3) + t                   # [..., N, 3]
    cov_w = Rinv.transpose(-2, -1) @ cov_c @ Rinv         # [..., N, 3, 3]
    return mu_w, cov_w


def invert_frame(frame: CanonicalFrame) -> CanonicalFrame:
    """
    Compute Φ_o⁻¹ = (R_oᵀ, -R_oᵀ t_o).

    Args:
        frame: CanonicalFrame  Φ_o = (R_o, t_o)

    Returns:
        CanonicalFrame  Φ_o⁻¹
    """
    R_inv = frame.R_w2c.transpose(-1, -2)
    t_inv = -torch.einsum('...ij,...j->...i', R_inv, frame.t_w2c)
    return CanonicalFrame(R_w2c=R_inv, t_w2c=t_inv)


def compose_frames(
    frame_a: CanonicalFrame,
    frame_b: CanonicalFrame,
) -> CanonicalFrame:
    """
    Compose two SE(3) frames:  Φ_a ∘ Φ_b.

    R_ab = R_a R_b
    t_ab = R_a t_b + t_a

    Args:
        frame_a, frame_b: CanonicalFrame

    Returns:
        CanonicalFrame
    """
    R_ab = frame_a.R_w2c @ frame_b.R_w2c
    t_ab = torch.einsum('...ij,...j->...i', frame_a.R_w2c, frame_b.t_w2c) + frame_a.t_w2c
    return CanonicalFrame(R_w2c=R_ab, t_w2c=t_ab)


# ═══════════════════════════════════════════════════════════════════════════
# §8  Cross-object transfer (used by Executor + cross-object equivariance loss)
# ═══════════════════════════════════════════════════════════════════════════

def transfer_action_frames(
    frame_source: CanonicalFrame,
    frame_target: CanonicalFrame,
) -> Tuple[CanonicalFrame, CanonicalFrame]:
    """
    Compute the pre/post transforms needed to transfer an action
    from source object to target object via canonical space conjugation:

        E_{target}(g) ≈ Φ_target⁻¹ ∘ E_canon(g) ∘ Φ_target

    Args:
        frame_source: Φ_{o_A}  (source object canonical frame)
        frame_target: Φ_{o_B}  (target object canonical frame)

    Returns:
        pre_transform:  Φ_target     (world → canonical before execution)
        post_transform: Φ_target⁻¹   (canonical → world after execution)
    """
    pre_transform = frame_target
    post_transform = invert_frame(frame_target)
    return pre_transform, post_transform


def conjugate_action(
    positions: torch.Tensor,
    covariances: torch.Tensor,
    frame: CanonicalFrame,
    executor_fn,
    action_params: dict,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full cross-object transfer pipeline:
      1. Map target object → canonical space
      2. Apply action in canonical space
      3. Map back to world space

    Args:
        positions:     [B, N, 3]      target object Gaussian centers (world)
        covariances:   [B, N, 3, 3]   target object covariances (world)
        frame:         CanonicalFrame  Φ_target
        executor_fn:   callable        E_canon(positions, covariances, **action_params)
                                       returns (positions', covariances')
        action_params: dict            action parameters (ℓ, h, ξ, ρ parsed from token)

    Returns:
        positions_world:   [B, N, 3]      updated positions (world)
        covariances_world: [B, N, 3, 3]   updated covariances (world)
    """
    # Step 1: World → Canonical
    pos_canon, cov_canon = to_canonical(positions, covariances, frame)

    # Step 2: Execute in canonical space
    pos_acted, cov_acted = executor_fn(pos_canon, cov_canon, **action_params)

    # Step 3: Canonical → World
    pos_world, cov_world = from_canonical(pos_acted, cov_acted, frame)

    return pos_world, cov_world



# ═══════════════════════════════════════════════════════════════════════════
# §9  Distance metrics (for L_clos / L_inv / L_comm losses)
# ═══════════════════════════════════════════════════════════════════════════

def frame_distance(
    frame_a: CanonicalFrame,
    frame_b: CanonicalFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distance between two SE(3) frames.

    Returns:
        rot_dist:   [B]   geodesic rotation distance (angle in radians)
        trans_dist: [B]   Euclidean translation distance
    """
    # Rotation distance: angle of R_a R_b^T
    R_diff = frame_a.R_w2c @ frame_b.R_w2c.transpose(-1, -2)
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    rot_dist = torch.acos(((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))

    # Translation distance
    trans_dist = (frame_a.t_w2c - frame_b.t_w2c).norm(dim=-1)

    return rot_dist, trans_dist


def gaussian_set_distance(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mean per-Gaussian center displacement (scene-space distance d_M).
    Used for closure / inverse / commutator losses.

    Args:
        pos_a: [..., N, 3]
        pos_b: [..., N, 3]
        mask:  optional [..., N] bool — True = include in mean.
               If None, uses vanilla mean (assumes no padding).

    Returns:
        dist: [...] mean Euclidean distance across (valid) Gaussians.
    """
    diff = (pos_a - pos_b).norm(dim=-1)                      # [..., N]
    if mask is None:
        return diff.mean(dim=-1)
    return masked_mean(diff, mask, dim=-1)
