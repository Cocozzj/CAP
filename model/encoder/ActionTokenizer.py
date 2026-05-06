"""
Action Tokenizer (Stage 3).

    z_motion [B, T_act, K, motion_dim]
      → 4 sub-projections  (ℓ, h, ξ, ρ)
      → concat into structured token  g = (ℓ, h, ξ, ρ)
      → single unified VQ codebook   K=512, dim=token_dim
      → Fusion MLP         (token_dim → motion_dim)
      → Decoder MLP        (reconstruct z_motion for training)
      → Physical heads     (auxiliary SE(3) / physics param supervision)

    Codebook initialization (structured by sub-dimension):
        [0 : d_l)            ℓ-section : 3D translation grid (physical dims 0:3)
        [d_l : d_l+d_h)      h-section : cube24 rotation matrices (physical dims 0:9)
        [d_l+d_h : -d_rho)   ξ-section : random (so(3) small ball)
        [-d_rho : ]           ρ-section : random (physics params)

    Physics plugin toggle:
        enabled=true  → full g = (ℓ, h, ξ, ρ)
        enabled=false → rigid g = (ℓ, h, ξ, 0)  with zero-padded ρ slot
        Same fusion MLP weights in both cases → Stage-0 → Stage-1 compatible.
"""
from __future__ import annotations
from typing import Dict, Any
import itertools
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ===========================================================================
# Cube24: chiral octahedral rotation group
# ===========================================================================
def generate_cube24() -> torch.Tensor:
    """
    Generate 24 rotation matrices of the chiral octahedral group O ≅ S₄.

    These are all 3×3 signed permutation matrices with determinant +1,
    i.e. the rotational symmetries of a cube / regular octahedron.

    Returns:
        [24, 3, 3] rotation matrices
    """
    matrices = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            M = torch.zeros(3, 3)
            for i, (j, s) in enumerate(zip(perm, signs)):
                M[i, j] = s
            if torch.det(M) > 0.5:                             # det = +1
                matrices.append(M)
    assert len(matrices) == 24, f"Expected 24, got {len(matrices)}"
    return torch.stack(matrices)                                # [24, 3, 3]


# ===========================================================================
# Translation grid
# ===========================================================================
def generate_translation_grid(
    delta_step: float, max_steps: int, K: int
) -> torch.Tensor:
    """
    Generate K representative 3D translations from a discrete lattice.

    Lattice spans [-max_steps*δ, +max_steps*δ] per axis.
    Sorted by distance from origin — small translations are more common
    in practice and should occupy lower codebook indices.

    Returns:
        [K, 3] translation vectors
    """
    steps = torch.arange(-max_steps, max_steps + 1).float() * delta_step
    grid = torch.stack(
        torch.meshgrid(steps, steps, steps, indexing="ij"), dim=-1
    )
    grid = grid.reshape(-1, 3)                                  # [N, 3]

    # Sort by L2 distance from origin (small translations first)
    dists = grid.norm(dim=-1)
    _, indices = dists.sort()
    return grid[indices[:K]]                                    # [K, 3]


# ===========================================================================
# Vector Quantizer (straight-through estimator)
# ===========================================================================
class VectorQuantizer(nn.Module):
    """
    Discrete bottleneck with straight-through gradient estimator.

    Forward:  hard assignment (argmin distance)
    Backward: gradients bypass quantization via straight-through
    """

    def __init__(
        self,
        num_codes: int,
        dim: int,
        beta: float = 0.25,
        restart_interval: int = 1000,
        restart_threshold: float = 0.1,
    ):   
        """
        Args:
            num_codes: codebook size K
            dim:       embedding dimension
            beta:      commitment loss weight (input → codebook alignment)
        """
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.beta = beta
        self.restart_interval = restart_interval
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
        
        self.register_buffer("usage_ema", torch.zeros(num_codes))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def l2_distances(self, flat: torch.Tensor) -> torch.Tensor:
        """
        Squared L2 distances from each row of flat to every codebook entry.

        Args:
            flat: [N, dim]
        Returns:
            dist: [N, num_codes]
        """
        return (
            flat.pow(2).sum(dim=-1, keepdim=True)   # [N, 1]
            + self.codebook.weight.pow(2).sum(dim=-1)  # [num_codes]
            - 2.0 * flat @ self.codebook.weight.T   # [N, num_codes]
        )

    def encode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Nearest-neighbour lookup with no loss computation.

        Args:
            z: [..., dim]
        Returns:
            indices:  [...] long   — codebook indices
            min_dist: [...] float  — distance to nearest entry (for logging)
        """
        orig_shape = z.shape
        flat = z.reshape(-1, self.dim)
        dist = self.l2_distances(flat)               # [N, num_codes]
        min_vals, indices = dist.min(dim=-1)         # [N], [N]
        return (
            indices.reshape(orig_shape[:-1]),
            min_vals.reshape(orig_shape[:-1]),
        )

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [..., dim] continuous features

        Returns:
            quantized:  [..., dim]  (straight-through)
            indices:    [...]       codebook indices
            vq_loss:    scalar      (codebook_loss + β * commit_loss)
        """
        orig_shape = z.shape
        flat = z.reshape(-1, self.dim)                          # [N, dim]

        # ── Nearest neighbor ──
        dist = self.l2_distances(flat)                          # [N, K]
        indices = dist.argmin(dim=-1)                           # [N]
        quantized = self.codebook(indices).reshape(orig_shape)  # [..., dim]

        
        # ─── Dead code restart（只在训练时触发，不影响 self.encode） ───
        if self.training:
            with torch.no_grad():
                usage = torch.bincount(indices, minlength=self.num_codes).float()
                decay = 0.99
                self.usage_ema.mul_(decay).add_(usage * (1 - decay))
                self.step_count += 1

                should_check = (self.step_count % self.restart_interval == 0).item()  # 仍然同步，但更明确
                if should_check:
                    avg_usage = self.usage_ema.mean()
                    dead_mask = self.usage_ema < (avg_usage * self.restart_threshold)
                    n_dead = int(dead_mask.sum().item())
                    if n_dead > 0 and flat.shape[0] >= n_dead:
                        rand_idx = torch.randint(
                            0, flat.shape[0], (n_dead,), device=flat.device
                        )
                        # Cast to codebook dtype: under AMP, ``flat`` is fp16
                        # but ``codebook.weight`` is fp32 (trainable params
                        # are kept in master precision).  Index assignment
                        # requires matching dtypes, otherwise raises
                        # "Index put requires the source and destination dtypes match".
                        # Triggers ~every ``restart_interval`` steps when small K
                        # produces dead codes (esp. K=64 ablation).
                        self.codebook.weight.data[dead_mask] = (
                            flat[rand_idx].detach().to(self.codebook.weight.dtype)
                        )
                        self.usage_ema[dead_mask] = avg_usage
        
        # ── Losses ──
        codebook_loss = F.mse_loss(quantized, z.detach())
        commit_loss = F.mse_loss(z, quantized.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # ── Straight-through estimator ──
        quantized_st = z + (quantized - z).detach()

        indices = indices.reshape(orig_shape[:-1])              # [...]
        return quantized_st, indices, vq_loss

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """Codebook lookup by index. indices: [...] → [..., dim]"""
        return self.codebook(indices)


# ===========================================================================
# ActionTokenizer
# ===========================================================================
class ActionTokenizer(nn.Module):
    """
    Stage 3: Quantize continuous motion features into structured action tokens.

    Each codebook entry is a complete action token g = (ℓ, h, ξ, ρ):
        ℓ  ∈ ℝ^d_l : translation embedding   (physical dims 0:3 = xyz)
        h  ∈ ℝ^d_h : rotation embedding       (physical dims 0:9 = flat R)
        ξ  ∈ ℝ^d_xi: micro-rotation embedding (physical dims 0:3 = so(3))
        ρ  ∈ ℝ^d_rho: deformation embedding   (zero-padded when disabled)

    action codebook: K_codes entries × token_dim dimensions.
    (K_obj = number of object slots; K_codes = codebook size — kept separate below.)
    """

    def __init__(self, tokenizer_param: dict):
        super().__init__()

        # ── Global dims ──
        in_dim = int(tokenizer_param.get("in_dim", 128))
        num_codes = int(tokenizer_param.get("num_action_codebook", 512))
        beta = float(tokenizer_param.get("beta", 0.25))
        restart_interval = int(tokenizer_param.get("vq_restart_interval", 1000))
        restart_threshold = float(tokenizer_param.get("vq_restart_threshold", 0.1))

        # ── Sub-component configs ──
        trans_cfg = tokenizer_param.get("translation", {})
        rot_cfg = tokenizer_param.get("rotation", {})
        micro_cfg = tokenizer_param.get("micro_rotation", {})
        deform_cfg = tokenizer_param.get("deformation", {})
        
        token_dim = int(tokenizer_param.get("token_dim", 64))
        sub_token_dim = token_dim // 4

        # max_angle in degrees → convert to radians for so(3) norm constraint
        xi_max_angle_deg = float(micro_cfg.get("max_angle", 5.0))
        self.xi_max_norm = xi_max_angle_deg * (math.pi / 180.0)  # ≈ 0.0873 rad
        
        d_l = int(trans_cfg.get("dim", sub_token_dim))
        d_h = int(rot_cfg.get("dim", sub_token_dim))
        d_xi = int(micro_cfg.get("dim", sub_token_dim))
        d_rho = int(deform_cfg.get("dim", sub_token_dim))
        
        # Validate: sub-dims must sum exactly to token_dim
        actual_token_dim = d_l + d_h + d_xi + d_rho
        if actual_token_dim != token_dim:
            raise ValueError(
                f"Sub-dim sum d_l({d_l}) + d_h({d_h}) + d_xi({d_xi}) + d_rho({d_rho}) "
                f"= {actual_token_dim} != token_dim={token_dim}. "
                f"Either set token_dim to a multiple of 4, or explicitly set each "
                f"sub-dim in the translation/rotation/micro_rotation/deformation config."
            )

        self.use_rho = bool(deform_cfg.get("enabled", True))
        self.in_dim = in_dim
        self.token_dim = token_dim
        self.d_l = d_l
        self.d_h = d_h
        self.d_xi = d_xi
        self.d_rho = d_rho

        # Sub-dim slicing offsets
        self._h_start = d_l
        self._xi_start = d_l + d_h
        self._rho_start = d_l + d_h + d_xi

        # ── Sub-projections: in_dim → sub-dims ──
        self.proj_l = nn.Linear(in_dim, d_l)
        self.proj_h = nn.Linear(in_dim, d_h)
        self.proj_xi = nn.Linear(in_dim, d_xi)
        self.proj_rho = nn.Linear(in_dim, d_rho) if self.use_rho else None

        # ── Action VQ codebook ──
        self.vq = VectorQuantizer(num_codes, token_dim, beta, restart_interval, restart_threshold)

        # ── Physical initialization ──
        self._init_codebook(trans_cfg)

        # ── Fusion MLP: token_dim → in_dim ──
        self.fusion = nn.Sequential(
            nn.Linear(token_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        # ── Decoder MLP: reconstruct z_motion for training ──
        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )

        # ── Physical param heads ──────────────────────────────────────
        # ℓ / h / ξ each have a small projection head that turns the sub-token
        # into the canonical physical parameterisation (translation, 6D rot,
        # so(3) twist).
        # ρ is DIFFERENT: per Physics Plugin PDF §1.3, ρ is a structured tuple
        # of named physical parameters (E, ν, ρ_m, F, μ, damping, dt) — NOT a
        # latent embedding.  So d_rho is set to match the required slot count
        # (currently 9, see RhoParser.RHO_DIM) and ``q_rho`` IS the output —
        # no extra head is needed.  RhoParser slices it directly.
        self.head_l  = nn.Linear(d_l, 3)                        # → ℝ³ translation
        self.head_h  = nn.Linear(d_h, 6)                        # → 6D + Gram-Schmidt
        self.head_xi = nn.Linear(d_xi, 3)                       # → so(3) vector
        # NOTE: head_rho was previously here as auxiliary supervision; removed
        # because q_rho now directly carries the named physics slots.

    # ---------------------------------------------------------------
    # Physical codebook initialization
    # ---------------------------------------------------------------
    def _init_codebook(self, trans_cfg: dict):
        """
        Initialize unified codebook with physical structure in sub-dimensions.

        ℓ-section [0:d_l]:           first 3 dims = translation grid, rest ≈ 0
        h-section [d_l:d_l+d_h]:     first 9 dims = cube24 rotation (cycled), rest ≈ 0
        ξ-section [_xi_start:_rho_start]: small random (learned from data)
        ρ-section [_rho_start:]:      small random (learned from data)
        """
        K = self.vq.num_codes
        dev = self.vq.codebook.weight.device
        dtype = self.vq.codebook.weight.dtype

        init = torch.randn(K, self.token_dim, device=dev, dtype=dtype) * 0.01

        # ── ℓ: translation grid → first 3 physical dims ──
        delta_step = float(trans_cfg.get("delta_step", 0.05))
        max_steps = int(trans_cfg.get("max_steps", 5))
        full_lattice = (2 * max_steps + 1) ** 3
        if K > full_lattice:
            raise ValueError(
                f"Codebook K={K} exceeds translation lattice size "
                f"{full_lattice} (max_steps={max_steps}). "
                f"Increase max_steps or reduce num_codes."
            )
        if K < full_lattice:
            warnings.warn(
                f"Codebook K={K} < lattice size {full_lattice}. "
                f"Only the {K} smallest translations are initialized. "
                f"Large displacements will not be seeded in the codebook.",
                stacklevel=2,
            )
        translations = generate_translation_grid(delta_step, max_steps, K)
        translations = translations.to(dev, dtype)
        n_trans_dims = min(3, self.d_l)
        init[:, :n_trans_dims] = translations[:, :n_trans_dims]  # physical ℓ dims

        # ── h: cube24 rotations → first 9 physical dims (initialized cyclically;
        #    duplicates resolved via dead code restart during training) ──
        rotations = generate_cube24().to(dev, dtype)            # [24, 3, 3]
        rot_flat = rotations.reshape(24, 9)                     # [24, 9]
        n_rot_dims = min(9, self.d_h)
        for i in range(K):
            init[i, self._h_start:self._h_start + n_rot_dims] = (
                rot_flat[i % 24, :n_rot_dims]
            )

        self.vq.codebook.weight.data.copy_(init)

    # ---------------------------------------------------------------
    # Sub-dimension slicing helpers
    # ---------------------------------------------------------------
    def _split_token(self, token: torch.Tensor):
        """Split [..., token_dim] → (q_l, q_h, q_xi, q_rho)."""
        q_l = token[..., :self.d_l]
        q_h = token[..., self._h_start:self._xi_start]
        q_xi = token[..., self._xi_start:self._rho_start]
        q_rho = token[..., self._rho_start:]
        return q_l, q_h, q_xi, q_rho
    
    # ---------------------------------------------------------------
    # Gram-Schmidt rotation matrix
    # ---------------------------------------------------------------
    @staticmethod
    def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
        """
        6D representation → 3×3 rotation matrix via Gram-Schmidt.
        Reference: Zhou et al. "On the Continuity of Rotation Representations
        in Neural Networks", CVPR 2019.
        
        Args:
            rot6d: [..., 6]
        Returns:
            R:     [..., 3, 3]   valid SO(3) rotation matrix
        """
        a1 = rot6d[..., :3]
        a2 = rot6d[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)

    # ---------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------
    def forward(self, z_motion: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            z_motion: [B, T_act, K, motion_dim]  from MotionEncoder

        Returns:
            tokens:          [B, T_act, K]       codebook indices
            quantized:       [B, T_act, K, motion_dim]  (fused, straight-through)
            vq_loss:         scalar
            recon:           [B, T_act, K, motion_dim]  (decoder output for training.py)
            sub_quantized:   {"l", "h", "xi", "rho"} each [B, T_act, K, d_sub]
            physical_params: {"translation": [..,3], "rotation": [..,9],
                              "micro_rotation": [..,3], "deformation": [..,n]}
        """
        B, T, K_obj, D = z_motion.shape

        # ── 1. Project to sub-spaces ──
        z_l = self.proj_l(z_motion)                             # [B, T, K, d_l]
        z_h = self.proj_h(z_motion)                             # [B, T, K, d_h]
        z_xi = self.proj_xi(z_motion)                           # [B, T, K, d_xi]

        # ── 2. ρ branch: physics plugin toggle ──
        if self.use_rho:
            z_rho = self.proj_rho(z_motion)                     # [B, T, K, d_rho]
        else:
            z_rho = z_motion.new_zeros(B, T, K_obj, self.d_rho)

        # ── 3. Concat → structured token → single VQ ──
        z_token = torch.cat([z_l, z_h, z_xi, z_rho], dim=-1)   # [B, T, K, token_dim]
        quantized_token, indices, vq_loss = self.vq(z_token)    # [B, T, K, token_dim], [B, T, K], scalar

        # ── 4. Split quantized back into sub-components ──
        q_l, q_h, q_xi, q_rho = self._split_token(quantized_token)

        # ── 5. Fuse → motion_dim ──
        fused = self.fusion(quantized_token)                    # [B, T, K, motion_dim]

        # ── 6. Decode (reconstruction target computed in training.py) ──
        recon = self.decoder(fused)                             # [B, T, K, motion_dim]

        # ── 7. Physical param heads (always computed, loss external) ──
        pred_translation = self.head_l(q_l)                     # [B, T, K, 3]
        pred_rotation_6d = self.head_h(q_h)                     # [B, T, K, 6]
        pred_rotation_mat = self._rot6d_to_matrix(pred_rotation_6d)  # [B, T, K, 3, 3]
        pred_rotation = pred_rotation_mat.reshape(*pred_rotation_mat.shape[:-2], 9)            # [B, T, K, 9]，但保证是 SO(3)
        pred_micro_rot_raw = self.head_xi(q_xi)                     # [B, T, K, 3]
        # 软约束：||ξ|| < xi_max_norm，用 tanh 平滑
        direction = F.normalize(pred_micro_rot_raw, dim=-1, eps=1e-8)
        norm = pred_micro_rot_raw.norm(dim=-1, keepdim=True)  # 仍然要拿 norm 算 magnitude
        magnitude = self.xi_max_norm * torch.tanh(norm / self.xi_max_norm)
        pred_micro_rot = direction * magnitude
        
        # ρ: q_rho IS the named physics tuple — feed it directly to the
        # executor (RhoParser will slice it per RHO_DIM=9 slots).
        # When use_rho=False (physics plugin disabled), we surface a zero
        # tensor so downstream shape checks still pass.
        pred_deform = q_rho if self.use_rho else None

        return {
            "tokens": indices,                                  # [B, T, K]
            "quantized": fused,                                 # [B, T, K, motion_dim]
            "vq_loss": vq_loss,
            "recon": recon,                                     # [B, T, K, motion_dim]
            "sub_quantized": {
                "l": q_l,                                       # [B, T, K, d_l]
                "h": q_h,                                       # [B, T, K, d_h]
                "xi": q_xi,                                     # [B, T, K, d_xi]
                "rho": q_rho,                                   # [B, T, K, d_rho]
            },
            "physical_params": {
                "translation": pred_translation,                # [B, T, K, 3]
                "rotation": pred_rotation,                      # [B, T, K, 9]
                "micro_rotation": pred_micro_rot,               # [B, T, K, 3]
                "deformation": pred_deform,                     # [B, T, K, n] or None
            },
        }

    # ---------------------------------------------------------------
    # Inference helpers
    # ---------------------------------------------------------------
    @torch.no_grad()
    def encode(self, z_motion: torch.Tensor) -> torch.Tensor:
        """
        Encode motion features to token indices only (no losses, no decoder).

        Args:
            z_motion: [B, T_act, K, motion_dim]

        Returns:
            indices: [B, T_act, K]
        """
        z_l = self.proj_l(z_motion)
        z_h = self.proj_h(z_motion)
        z_xi = self.proj_xi(z_motion)

        if self.use_rho:
            z_rho = self.proj_rho(z_motion)
        else:
            z_rho = z_motion.new_zeros(*z_motion.shape[:-1], self.d_rho)

        z_token = torch.cat([z_l, z_h, z_xi, z_rho], dim=-1)

        # Delegate to VectorQuantizer.encode() — single source of truth for distance
        indices, _ = self.vq.encode(z_token)
        return indices

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decode token indices back to fused quantized features (for executor).

        Args:
            tokens: [B, T_act, K]  codebook indices

        Returns:
            quantized: [B, T_act, K, motion_dim]
        """
        quantized_token = self.vq.lookup(tokens)                # [..., token_dim]
        return self.fusion(quantized_token)                     # [..., motion_dim]

    def tokens_to_physical_params(self, tokens: torch.Tensor) -> Dict[str, Any]:
        """Decode token indices to structured physical_params for the Executor.

        Inverse of the forward path's "VQ → split → physical heads" sub-pipeline.
        Used at inference time when Planner outputs only token indices but
        the Executor needs structured (translation, rotation, micro_rotation,
        deformation) inputs.

        Args:
            tokens: [B, T_act, K] long — codebook indices

        Returns:
            dict with the same keys/shapes as ActionTokenizer.forward()'s
            ``physical_params``:
                translation:    [B, T_act, K, 3]
                rotation:       [B, T_act, K, 9]    (valid SO(3), via Gram-Schmidt)
                micro_rotation: [B, T_act, K, 3]    (||·|| ≤ xi_max_norm)
                deformation:    [B, T_act, K, n] or None
        """
        # Lookup quantised token from codebook → split into sub-components
        quantized_token = self.vq.lookup(tokens)                # [..., token_dim]
        q_l, q_h, q_xi, q_rho = self._split_token(quantized_token)

        # ── Translation ──
        pred_translation = self.head_l(q_l)                     # [..., 3]

        # ── Rotation: 6D → SO(3) via Gram-Schmidt (consistent with forward) ──
        pred_rotation_6d = self.head_h(q_h)                     # [..., 6]
        pred_rotation_mat = self._rot6d_to_matrix(pred_rotation_6d)  # [..., 3, 3]
        pred_rotation = pred_rotation_mat.reshape(
            *pred_rotation_mat.shape[:-2], 9
        )                                                       # [..., 9]

        # ── Micro-rotation with tanh-clamped norm (||·|| ≤ xi_max_norm) ──
        pred_micro_rot_raw = self.head_xi(q_xi)                 # [..., 3]
        direction = F.normalize(pred_micro_rot_raw, dim=-1, eps=1e-8)
        norm = pred_micro_rot_raw.norm(dim=-1, keepdim=True)
        magnitude = self.xi_max_norm * torch.tanh(norm / self.xi_max_norm)
        pred_micro_rot = direction * magnitude

        # ── Deformation (None when physics plugin is disabled) ──
        # ρ is the named physics tuple itself (Physics-Plugin PDF §1.3, see
        # ``model/executor/deform/rho_parser.py`` for slot map).  q_rho IS the
        # output — no head_rho projection needed.  Mirrors the forward path.
        pred_deform = q_rho if self.use_rho else None

        return {
            "translation":    pred_translation,
            "rotation":       pred_rotation,
            "micro_rotation": pred_micro_rot,
            "deformation":    pred_deform,
        }