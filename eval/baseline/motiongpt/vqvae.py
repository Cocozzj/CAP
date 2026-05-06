"""Per-frame VQ-VAE for pose deltas — self-contained, MotionGPT-internal.

This is a small motion tokenizer used to convert continuous pose deltas
([T-1, 7] per trajectory) into a discrete sequence of motion-token IDs
([T-1] integers in [0, K)) that T5 can be fine-tuned on.

Design goals
------------
* Self-contained — no dependency on the project's main model.* code.
* Tiny — ~50K params, trains in minutes on 4× A100.
* Per-frame — the "flat" structure means each delta is encoded
  independently, mirroring MotionGPT's original (no temporal dilation).

Architecture
------------
Encoder    : Linear(7 → hidden) → SiLU → Linear(hidden → code_dim)
Quantizer  : VectorQuantizer(K, code_dim) with straight-through grad
Decoder    : Linear(code_dim → hidden) → SiLU → Linear(hidden → 7)

Loss
----
L = MSE(recon, target)
  + β · MSE(z_e, sg(z_q))           # commitment loss
  + MSE(sg(z_e), z_q)               # codebook loss (no EMA — simple SGD update)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Vector quantizer (straight-through, no EMA)
# ──────────────────────────────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    def __init__(self, K: int, code_dim: int, beta: float = 0.25):
        super().__init__()
        self.K, self.code_dim, self.beta = K, code_dim, float(beta)
        self.codebook = nn.Embedding(K, code_dim)
        # initialise codes to small random vectors
        self.codebook.weight.data.uniform_(-1.0 / K, 1.0 / K)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: int [..., ] → vectors [..., code_dim]"""
        return self.codebook(ids)

    def forward(self, z_e: torch.Tensor):
        """z_e: [B, T, code_dim] → (z_q [B,T,code_dim], ids [B,T], loss scalar)"""
        B, T, C = z_e.shape
        z_flat = z_e.reshape(-1, C)                         # [BT, C]
        # Squared distance between every input and every codebook entry
        dist = (z_flat.pow(2).sum(1, keepdim=True)
                - 2 * z_flat @ self.codebook.weight.T
                + self.codebook.weight.pow(2).sum(1))       # [BT, K]
        ids_flat = dist.argmin(dim=1)                        # [BT]
        z_q_flat = self.codebook(ids_flat)                   # [BT, C]

        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        commit_loss   = F.mse_loss(z_q_flat.detach(), z_flat)
        loss = codebook_loss + self.beta * commit_loss

        # Straight-through: pass gradient from z_q back to z_e
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()
        z_q  = z_q_flat.reshape(B, T, C)
        ids  = ids_flat.reshape(B, T)
        return z_q, ids, loss


# ──────────────────────────────────────────────────────────────────────
# Flat VQ-VAE
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlatVQVAEArgs:
    in_dim:    int = 7
    hidden:    int = 128
    code_dim:  int = 32
    K:         int = 64
    beta:      float = 0.25


class FlatVQVAE(nn.Module):
    """Per-frame VQ-VAE on pose deltas (3 trans + 4 quat = 7 dims)."""

    def __init__(self, in_dim: int = 7, hidden: int = 128,
                 code_dim: int = 32, K: int = 64, beta: float = 0.25):
        super().__init__()
        self.in_dim   = in_dim
        self.hidden   = hidden
        self.code_dim = code_dim

        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, code_dim),
        )
        self.quantizer = VectorQuantizer(K=K, code_dim=code_dim, beta=beta)
        self.dec = nn.Sequential(
            nn.Linear(code_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_dim] → z_e [B, T, code_dim]"""
        return self.enc(x)

    @torch.no_grad()
    def encode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, in_dim] → ids [B, T] int"""
        z_e = self.encode(x)
        _, ids, _ = self.quantizer(z_e)
        return ids

    def decode_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: [B, T] int → recon [B, T, in_dim]"""
        z_q = self.quantizer.lookup(ids)
        return self.dec(z_q)

    # Alias — older infer.py call sites use ``decode_from_ids``.
    decode_from_ids = decode_ids

    def forward(self, x: torch.Tensor):
        """Full pass for training.  Returns (recon, ids, vq_loss)."""
        z_e = self.encode(x)
        z_q, ids, vq_loss = self.quantizer(z_e)
        recon = self.dec(z_q)
        return recon, ids, vq_loss


__all__ = ["FlatVQVAE", "FlatVQVAEArgs", "VectorQuantizer"]
