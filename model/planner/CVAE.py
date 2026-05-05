"""
CVAE — Conditional VAE for Task-Conditioned Sequence Generation (Stage 4.2)

                    Training                                    Inference
                    ────────                                    ─────────
GT token_indices ──→  Posterior ──→     z                       (no seq)
                            │           │                           │
                    (KL pulls it        │                       z ~ N(0,I)
                        toward N(0,I))  │                           │
                                        ▼                           ▼
            task_emb ──────────→ build_cond_mem         task_emb ──→ build_cond_mem
                                        │                           │
                                        ▼                           ▼
                                Decoder (teacher            Decoder (autoregressive
                                forcing, knows GT)          loop, generates new)
                                        │                           │
                                        ▼                           ▼
                                logits → CE loss            sequences → Executor
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions.AutoregressiveDecoder import AutoregressiveDecoder

class CVAE(nn.Module):
    """
    Conditional VAE  for task-conditioned primitive-sequence generation.

    Key design: condition memory (cond_mem) built from [task, z] is passed to the AR decoder as cross-attention memory. 
    """

    def __init__(
        self,
        *,
        K_prim: int,
        task_dim: int,
        d_model: int = 256,
        z_dim: int = 32,                    
        # posterior encoder config
        posterior_encoder_cfg: Optional[Dict] = None,
        # condition memory config
        cond_mem_cfg: Optional[Dict] = None,
        # AR decoder config
        ardecoder_cfg: Optional[Dict] = None,
        # embedding tying
        tie_prim_embeddings: bool = False,
        prim_embed_weight: Optional[nn.Embedding] = None,
    ):
        super().__init__()
        # ── dimensions ─────────────────────────────────────────────
        self.K_prim = int(K_prim)
        self.task_dim = int(task_dim)
        self.d_model = int(d_model)
        self.z_dim = int(z_dim)
        self.tie_prim_embeddings = bool(tie_prim_embeddings)

        # ── config defaults ────────────────────────────────────────
        pe_cfg = posterior_encoder_cfg or {}
        cm_cfg = cond_mem_cfg or {}
        ar_cfg = ardecoder_cfg or {}

        # posterior encoder hyperparams
        self.max_len = int(pe_cfg.get("max_len", 512))
        n_layers_post = int(pe_cfg.get("n_layers", 2))
        n_heads_post = int(pe_cfg.get("n_heads", 8))
        d_ff_post = int(pe_cfg.get("d_ff", 1024))
        dropout_post = float(pe_cfg.get("dropout", 0.1))
        self.pooling = str(pe_cfg.get("pooling", "cls")).lower()

        # condition memory hyperparams
        self.M_cond = int(cm_cfg.get("M_tokens", 8))
        self.cm_drop = float(cm_cfg.get("dropout", 0.1))
        self.use_film = bool(cm_cfg.get("use_film", False))

        # ── primitive token embedding (for posterior encoder) ──────
        if prim_embed_weight is not None:
            self.prim_embed = prim_embed_weight
            assert isinstance(self.prim_embed, nn.Embedding)
            assert self.prim_embed.num_embeddings == self.K_prim
            assert self.prim_embed.embedding_dim == self.d_model
            self._owns_prim_embed = False
        else:
            self.prim_embed = nn.Embedding(self.K_prim, self.d_model)
            nn.init.normal_(self.prim_embed.weight, mean=0.0, std=0.02)
            self._owns_prim_embed = True

        # ── positional embedding (shared for posterior) ────────────
        self.pos_embed = nn.Embedding(self.max_len + 1, self.d_model)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # ── explicit [CLS] token for pooling ───────────────────────
        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        else:
            self.cls_token = None

        # ── Transformer posterior encoder ──────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads_post,
            dim_feedforward=d_ff_post,
            dropout=dropout_post,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.posterior = nn.TransformerEncoder(enc_layer, num_layers=n_layers_post)
        self.post_norm = nn.LayerNorm(self.d_model)

        # ── pooling projection head ────────────────────────────────
        self.pool_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        if self.pooling == "attn-pool":
            self.pool_q = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.normal_(self.pool_q, mean=0.0, std=0.02)

        # ── fuse (seq_pool, c_task) → μ, log_var ──────────────────
        self.task_gate = nn.Sequential(
            nn.Linear(self.d_model + self.task_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.SiLU(),
        )
        self.to_mu = nn.Linear(self.d_model, self.z_dim)
        self.to_logvar = nn.Linear(self.d_model, self.z_dim)
        # init logvar bias slightly negative → initial σ ≈ 0.6
        nn.init.constant_(self.to_logvar.bias, -1.0)

        # ── condition memory: (task, z) → [B, M, D] ───────────────
        self.cond_proj = nn.Sequential(
            nn.Linear(self.task_dim + self.z_dim, self.d_model * self.M_cond),
            nn.SiLU(),
            nn.Dropout(self.cm_drop),
        )
        self.cond_ln = nn.LayerNorm(self.d_model)

        # ── optional FiLM modulation on cond_mem ───────────────────
        if self.use_film:
            self.film_head = nn.Sequential(
                nn.Linear(self.task_dim + self.z_dim, 2 * self.d_model),
                nn.SiLU(),
                nn.Linear(2 * self.d_model, 2 * self.d_model),
            )

        # ── AR decoder (cross-attention to cond_mem) ───────────────
        _ar_cfg = dict(ar_cfg)
        _ar_cfg.setdefault("K_prim", self.K_prim)
        _ar_cfg.setdefault("d_model", self.d_model)
        self.decoder = AutoregressiveDecoder(**_ar_cfg)

        # ── optional: tie posterior & decoder embeddings ───────────
        if self.tie_prim_embeddings:
            self.tie_with_decoder_embeddings_()

    # ──────────────────── internal helpers ────────────────────────

    def _pos_add(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Add positional encoding starting at `offset`."""
        B, L, _ = x.shape
        if L + offset > self.max_len + 1:
            raise ValueError(f"Seq length {L}+offset {offset} exceeds max_len+1={self.max_len + 1}")
        pos_ids = torch.arange(offset, offset + L, device=x.device).unsqueeze(0).expand(B, L)
        return x + self.pos_embed(pos_ids)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    # ════════════════════════════════════════════════════════════════
    # 1. Posterior Encoder q_φ(z | seq, c_task)
    # ════════════════════════════════════════════════════════════════

    def encode_posterior(
        self,
        token_indices: torch.Tensor,                    # [B, L] int64
        task_emb: torch.Tensor,                    # [B, task_dim]
        seq_mask: Optional[torch.Tensor] = None,     # [B, L] bool True=valid
    ) -> Dict[str, torch.Tensor]:
        """
        Encode ground-truth sequence + task into posterior distribution.

        token_indices [B, L] 

        Args:
            token_indices: [B, L] — ground-truth primitive token ids
            task_emb: [B, task_dim] — quantized task embedding
            seq_mask:   [B, L] bool — True=valid, False=padding

        Returns:
            mu:     [B, z_dim]
            logvar: [B, z_dim]
            z:      [B, z_dim]  (reparameterized sample)
        """
        assert token_indices.dim() == 2
        B, L = token_indices.shape
        assert task_emb.shape == (B, self.task_dim)

        safe_indices = token_indices.clamp(min=0)         # pad_id=-1 → 0 (masked out by seq_mask)
        x = self.prim_embed(safe_indices)                 # [B, L, D]

        # ── prepend [CLS] if using cls pooling ─────────────────────
        if self.pooling == "cls":
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)            # [B, L+1, D]
            x = self._pos_add(x, offset=0)
            src_pad_mask = None
            if seq_mask is not None:
                pad = ~seq_mask.bool()
                # CLS is always valid → prepend False (not padded)
                pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], dim=1)
                src_pad_mask = pad
        else:
            x = self._pos_add(x, offset=0)
            src_pad_mask = (~seq_mask.bool()) if seq_mask is not None else None

        # ── Transformer encoder ────────────────────────────────────
        h = self.posterior(x, src_key_padding_mask=src_pad_mask)
        h = self.post_norm(h)                         # [B, L', D]

        # ── pooling → fixed-length vector ──────────────────────────
        if self.pooling == "cls":
            h_pool = self.pool_proj(h[:, 0, :])
        elif self.pooling == "mean":
            if src_pad_mask is None:
                h_pool = self.pool_proj(h.mean(dim=1))
            else:
                valid = (~src_pad_mask).float()
                denom = valid.sum(dim=1, keepdim=True).clamp_min(1e-6)
                h_pool = self.pool_proj(
                    (h * valid.unsqueeze(-1)).sum(dim=1) / denom
                )
        elif self.pooling == "attn-pool":
            q = self.pool_q.expand(B, 1, -1)
            attn = torch.softmax((q @ h.transpose(1, 2)) / (self.d_model ** 0.5), dim=-1)
            if src_pad_mask is not None:
                mask = (~src_pad_mask).unsqueeze(1).float()
                attn = attn * mask
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            h_pool = self.pool_proj((attn @ h).squeeze(1))
        else:
            raise ValueError(f"Unknown pooling mode: {self.pooling}")

        # ── fuse with task → μ, log_var ────────────────────────────
        fused = self.task_gate(torch.cat([h_pool, task_emb], dim=-1))
        mu = self.to_mu(fused)
        logvar = self.to_logvar(fused)
        z = self._reparameterize(mu, logvar)

        return {"mu": mu, "logvar": logvar, "z": z}

    # ════════════════════════════════════════════════════════════════
    # 2. Prior Sampling
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def sample_prior(self, B: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Sample z ~ N(0, I) from the prior."""
        dev = device if device is not None else next(self.parameters()).device
        return torch.randn(B, self.z_dim, device=dev, dtype=next(self.parameters()).dtype)

    # ════════════════════════════════════════════════════════════════
    # 3. Condition Memory Builder
    # ════════════════════════════════════════════════════════════════

    def build_cond_mem(
        self,
        *,
        task_emb: torch.Tensor,                    # [B, task_dim]
        z_task: torch.Tensor,                        # [B, z_dim]
        extra_memories: Optional[List[Dict]] = None,  # extensible
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build condition memory for cross-attention.

        Constructs a memory bank that the AR decoder cross-attends to.
        Fixed block: task_emb + z_task → M_cond tokens.
        Optional: extra_memories for scene features, object embeddings, etc.

        Args:
            task_emb:     [B, task_dim]
            z_task:         [B, z_dim]
            extra_memories: list of {"x": [B,M_i,D], "mask": [B,M_i], "proj": Module}

        Returns:
            cond_mem:  [B, M_total, d_model]
            mem_mask:  [B, M_total] bool (True=valid)
        """
        assert task_emb.dim() == 2 and task_emb.shape[1] == self.task_dim
        assert z_task.dim() == 2 and z_task.shape[1] == self.z_dim
        B = task_emb.size(0)
        device = task_emb.device

        # ── fixed block: (task, z) → M_cond rich tokens ───────────
        tz = torch.cat([task_emb, z_task], dim=-1)         # [B, task_dim + z_dim]
        fixed = self.cond_proj(tz)                            # [B, d_model * M_cond]
        fixed = fixed.view(B, self.M_cond, self.d_model)      # [B, M_cond, D]

        # optional FiLM modulation
        if self.use_film:
            film = self.film_head(tz)                         # [B, 2 * d_model]
            gamma, beta = film.chunk(2, dim=-1)               # each [B, d_model]
            fixed = fixed * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        fixed = self.cond_ln(fixed)                           # [B, M_cond, D]

        mems = [fixed]
        masks = [torch.ones(B, self.M_cond, dtype=torch.bool, device=device)]

        # ── optional: variable-length extra memories ───────────────
        #   Each item must provide:
        #     "x":    [B, M_i, D'] — feature tensor
        #     "proj": nn.Module    — REQUIRED if D' != d_model (must be pre-registered)
        #     "mask": [B, M_i] bool (optional, default all-valid)
        if extra_memories:
            for item in extra_memories:
                x: torch.Tensor = item["x"]
                m: Optional[torch.Tensor] = item.get("mask")

                assert x.size(0) == B, "Extra memory batch mismatch"
                Mi = x.size(1)

                if x.size(-1) != self.d_model:
                    proj = item.get("proj")
                    if proj is None:
                        raise ValueError(
                            f"Extra memory has dim {x.size(-1)} != d_model {self.d_model} "
                            f"but no 'proj' module provided. Register the projection "
                            f"in __init__ and pass it via item['proj']."
                        )
                    x = proj(x)

                mems.append(x)
                masks.append(
                    m if m is not None
                    else torch.ones(B, Mi, dtype=torch.bool, device=device)
                )

        # ── concatenate ────────────────────────────────────────────
        cond_mem = torch.cat(mems, dim=1)                    # [B, M_total, D]
        mem_mask = torch.cat(masks, dim=1)                   # [B, M_total]

        return cond_mem, mem_mask

    # ════════════════════════════════════════════════════════════════
    # 4. Decode Wrappers (delegate to AR decoder)
    # ════════════════════════════════════════════════════════════════

    def decode_teacher_forcing(
        self,
        token_indices: torch.Tensor,                    # [B, L-1] left-shifted action tokens
        cond_mem: torch.Tensor,                      # [B, M, D]
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced decoding via AR decoder with cross-attention.

        Returns:
            logits: [B, L, K_prim]
            hidden: [B, L, D]
        """
        assert token_indices.dim() == 2
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        return self.decoder.forward_teacher_forcing(
            prev_tokens=token_indices,
            cond_mem=cond_mem,
            mem_mask=mem_mask,
        )

    def decode_scheduled_sampling(
        self,
        token_indices: torch.Tensor,                    # [B, L-1] left-shifted action tokens
        cond_mem: torch.Tensor,                      # [B, M, D]
        sample_prob: float = 0.0,
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Scheduled-sampling decoding: mix GT and own predictions during training.

        When sample_prob=0.0, equivalent to decode_teacher_forcing.

        Returns:
            logits: [B, L, K_prim]
            hidden: [B, L, D]
        """
        assert token_indices.dim() == 2
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        return self.decoder.forward_scheduled_sampling(
            prev_tokens=token_indices,
            cond_mem=cond_mem,
            sample_prob=sample_prob,
            mem_mask=mem_mask,
        )

    @torch.no_grad()
    def decode_generate(
        self,
        cond_mem: torch.Tensor,
        sampling_cfg: Dict[str, Any],
        mem_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Autoregressive generation via AR decoder with cross-attention.

        Returns:
            sequences: [B, L_out]
            gen_aux:   dict
        """
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        return self.decoder.generate(
            cond_mem=cond_mem,
            sampling_cfg=sampling_cfg,
            mem_mask=mem_mask,
        )

    # ════════════════════════════════════════════════════════════════
    # Optional: embedding tying
    # ════════════════════════════════════════════════════════════════

    def tie_with_decoder_embeddings_(self) -> None:
        """Tie posterior prim_embed weights with decoder token_embed."""
        if not self.tie_prim_embeddings:
            return
        if not hasattr(self.decoder, "token_embed"):
            raise AttributeError("Decoder has no 'token_embed' to tie with.")
        if self.decoder.token_embed.weight.shape != self.prim_embed.weight.shape:
            raise ValueError("Embedding shapes do not match for tying.")
        self.prim_embed.weight = self.decoder.token_embed.weight
        self._owns_prim_embed = False

