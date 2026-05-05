"""
TaskTokenizer: Action Token Sequence → Discrete Task Token

Action Token Sequence{c_prim,1:T} → embed → project(d_model) → [CLS]+pos → Transformer(d_model) → pool(take [CLS] position) 
                → h_task [B, d_model]  ->  proj(task_dim) → VQ → task_emb [B, task_dim]

TRAINING:  encode(token_indices, atomic_codebook) → task_id, task_emb, h_task, vq_loss
           reconstruct(task_emb) → [task_dim]  Reconstructed h_task
INFERENCE: encode(token_indices, atomic_codebook) → task_id, task_emb, h_task, vq_loss≈0
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════
# Vector Quantizer (EMA codebook)
# ══════════════════════════════════════════════════════════════════
class _EMAVectorQuantizer(nn.Module):
    """
    Vector quantization with EMA codebook updates.
    """

    def __init__(
        self,
        num_entries: int = 128,    # J = size of task codebook
        embed_dim: int = 64,       # task_dim
        beta: float = 0.25,
        ema_decay: float = 0.99,
        restart_threshold: int = 2,
        restart_interval: int = 1000,
    ):
        super().__init__()
        self.num_entries = num_entries
        self.embed_dim = embed_dim
        self.beta = beta
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold
        self.restart_interval = restart_interval

        # Codebook: PUBLIC — accessed by LanguageEncoder and CVAE
        self.codebook = nn.Embedding(num_entries, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / num_entries, 1.0 / num_entries)

        # EMA state (not model parameters, but persistent across steps)
        self.register_buffer("ema_count", torch.zeros(num_entries))
        self.register_buffer("ema_weight", self.codebook.weight.clone())
        self.register_buffer("usage_count", torch.zeros(num_entries))
        self.register_buffer("step_count", torch.zeros(1, dtype=torch.long))

    def _ema_update(self, indices: torch.Tensor, encoded: torch.Tensor):
        """
        Update codebook entries via exponential moving average.

        Args:
            indices: [N] — which codebook entries were selected
            encoded: [N, embed_dim] — the encoder outputs that selected them
        """
        if not self.training:
            return

        onehot = F.one_hot(indices, self.num_entries).float()
        counts = onehot.sum(dim=0)
        sums = onehot.t() @ encoded

        self.ema_count.mul_(self.ema_decay).add_(counts, alpha=1 - self.ema_decay)
        self.ema_weight.mul_(self.ema_decay).add_(sums, alpha=1 - self.ema_decay)

        n = self.ema_count.sum()
        count_smooth = (
            (self.ema_count + 1e-5)
            / (n + self.num_entries * 1e-5)
            * n
        )
        self.codebook.weight.data.copy_(self.ema_weight / count_smooth.unsqueeze(1))

        self.usage_count.add_(counts)

    def restart_dead_entries(self, encoded: torch.Tensor):
        """
        Replace dead codebook entries with random encoder outputs.

        Args:
            encoded: [N, embed_dim] — pool of encoder outputs to sample from
        """
        dead = self.usage_count < self.restart_threshold
        num_dead = dead.sum().item()

        if num_dead > 0 and encoded.size(0) > 0:
            perm = torch.randperm(encoded.size(0), device=encoded.device)
            replacements = encoded[perm[:num_dead]]

            if replacements.size(0) < num_dead:
                # IMPORTANT: device=encoded.device — without it, torch.randint
                # defaults to CPU, indexing a CUDA tensor with a CPU index
                # forces an implicit sync (perf hit) and warns in newer
                # PyTorch versions.  Match the perm path above.
                extra_idx = torch.randint(
                    0, encoded.size(0),
                    (num_dead - replacements.size(0),),
                    device=encoded.device,
                )
                extra = encoded[extra_idx]
                replacements = torch.cat([replacements, extra], dim=0)

            self.codebook.weight.data[dead] = replacements
            self.ema_weight[dead] = replacements
            self.ema_count[dead] = 1.0

        self.usage_count.zero_()

    def forward(
        self, h: torch.Tensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous vectors to nearest codebook entries.

        Args:
            h: [B, embed_dim] — continuous task representations (h_task)

        Returns:
            indices:  [B] long — codebook indices (task_id)
            quantized: [B, embed_dim] — codebook embeddings (task_emb),
                       with straight-through gradient to h
            vq_loss:  scalar — codebook + commitment loss
        """
        dists = (
            h.pow(2).sum(dim=-1, keepdim=True)
            - 2 * h @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=-1, keepdim=True).t()
        )

        indices = dists.argmin(dim=-1)
        quantized = self.codebook(indices)

        self._ema_update(indices, h.detach())
        
        # ─── 自动 dead code restart ───
        if self.training:
            with torch.no_grad():
                self.step_count += 1
                if (self.step_count % self.restart_interval == 0).item():
                    self.restart_dead_entries(h.detach())  # 用当前 batch 的 h 做替换池

        # EMA VQ: only commitment loss (push encoder h toward codebook).
        # Codebook itself is updated via EMA (_ema_update), NOT gradient.
        vq_loss = self.beta * F.mse_loss(h, quantized.detach())

        quantized_st = h + (quantized - h).detach()

        return indices, quantized_st, vq_loss


# ══════════════════════════════════════════════════════════════════
# Sinusoidal Positional Encoding
# ══════════════════════════════════════════════════════════════════
class _SinusoidalPosEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════
# Task Tokenizer
# ══════════════════════════════════════════════════════════════════
class TaskTokenizer(nn.Module):
    """
    Compresses a sequence of atomic action tokens into a single discrete task token.

    Pipeline:
    seq_tokens [B, L]  → embed → project(d_model) → [CLS]+pos → Transformer(d_model)
                   → [CLS] pool → proj(task_dim) → VQ → task_id

    Public attributes:
      self.quantizer.codebook — nn.Embedding(J, task_dim)
    """

    def __init__(
        self,
        atomic_dim: int = 128,     # d dimension of atomic codebook embeddings
        d_model: int = 256,        # Transformer hidden dimension
        task_dim: int = 64,        # VQ codebook dimension (bottleneck)
        num_task_tokens: int = 128, # J = task codebook size
        max_seq_len: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        vq_beta: float = 0.25,
        vq_ema_decay: float = 0.99,
        task_restart_interval: int = 1000,
        task_restart_threshold: int = 2,
    ):
        super().__init__()
        self.atomic_dim = atomic_dim
        self.d_model = d_model
        self.task_dim = task_dim
        self.num_task_tokens = num_task_tokens

        # ── Step 1: Token projection (atomic_dim → d_model) ─────
        self.token_proj = nn.Linear(atomic_dim, d_model)

        # ── Step 2: [CLS] token + positional encoding ───────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_encoding = _SinusoidalPosEncoding(
            d_model=d_model,
            max_len=max_seq_len + 1,  # +1 for [CLS]
            dropout=dropout,
        )

        # ── Step 3: Transformer encoder (in d_model space) ──────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ── Step 4: Bottleneck projection (d_model → task_dim) ──
        self.proj = nn.Sequential(
            nn.Linear(d_model, task_dim),
            nn.LayerNorm(task_dim),
        )

        # ── Step 5: Vector quantizer (in task_dim space) ────────
        self.quantizer = _EMAVectorQuantizer(
            num_entries=num_task_tokens,
            embed_dim=task_dim,
            beta=vq_beta,
            ema_decay=vq_ema_decay,
            restart_interval=int(task_restart_interval),
            restart_threshold=int(task_restart_threshold),
        )
        # ── Step 6: Reconstruction decoder ───────────────────────
        self.task_decoder = nn.Sequential(
            nn.Linear(task_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, task_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in [self.token_proj, self.proj[0],
                    self.task_decoder[0], self.task_decoder[2]]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        
        # Special tokens: small random
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02) 
    
    @property
    def codebook(self) -> nn.Embedding:
        return self.quantizer.codebook

    def _build_padding_mask(
        self, token_indices: torch.Tensor, pad_id: int = -1
    ) -> Optional[torch.Tensor]:
        """
        Build attention mask: True = ignore position.

        Args:
            token_indices: [B, T] — may contain pad_id for variable-length sequences.
            pad_id: padding token index.

        Returns:
            mask: [B, T+1] bool — True for positions to ignore.
                  None if no padding detected.
        """
        if (token_indices == pad_id).any():
            token_pad = token_indices == pad_id
            cls_col = torch.zeros(
                token_pad.size(0), 1, dtype=torch.bool, device=token_pad.device
            )
            return torch.cat([cls_col, token_pad], dim=1)
        return None

    # ══════════════════════════════════════════════════════════════
    # TRAINING + INFERENCE
    # ══════════════════════════════════════════════════════════════
    def encode(
        self,
        token_indices,
        token_embeds: torch.Tensor,    
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Atomic token sequence → discrete task token.

        Args:
            token_indices:   [B, T] long — atomic token indices from Encoder Stage 3
            token_embeds:    token_embeds from atomic codebook
                        
        Returns:
            task_id:        [B] long       — discrete task token index
            task_emb:     [B, task_dim]  — quantized task embedding (straight-through)
            h_task:         [B, task_dim]  — pre-VQ continuous feature (for losses/)
            vq_loss:        scalar         — commitment + codebook loss
        """
        B= token_embeds.shape[0]

        # ── Step 1: Embed + project to d_model ───────────────────
        x = self.token_proj(token_embeds)                          # [B, T, d_model]

        # ── Step 2: Prepend [CLS] + positional encoding ──────────
        cls = self.cls_token.expand(B, -1, -1)                     # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                             # [B, T+1, d_model]
        x = self.pos_encoding(x)

        # ── Step 3: Transformer encoder ──────────────────────────
        padding_mask = self._build_padding_mask(token_indices)
        x = self.transformer(x, src_key_padding_mask=padding_mask) # [B, T+1, d_model]

        # ── Step 4: [CLS] pool → bottleneck projection ──────────
        h_pooled = x[:, 0, :]                                     # [B, d_model]
        h_task = self.proj(h_pooled)                               # [B, task_dim]

        # ── Step 5: Vector quantization ──────────────────────────
        task_id, task_emb, vq_loss = self.quantizer(h_task)

        return task_id, task_emb, h_task, vq_loss

    # ══════════════════════════════════════════════════════════════
    # Reconstruction
    # ══════════════════════════════════════════════════════════════

    def reconstruct(self, task_emb: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized task embedding back to atomic sequence estimate.
        Args:
            task_emb: [B, task_dim] — quantized task embedding 

        Returns:
            [B, task_dim] — reconstructed h_task
        """
        return self.task_decoder(task_emb)                          # [B, task_dim]
