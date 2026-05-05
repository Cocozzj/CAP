"""
AutoregressiveDecoder — Cross-Attention Conditioned Causal Decoder (Stage 4.3)

Architecture:
  nn.TransformerDecoder — each layer has:
    1. Causal self-attention over action token sequence
    2. Cross-attention to condition memory (cond_mem from CVAE)
    3. Feed-forward sublayer

  cond_mem [B, M, D] contains projected task embedding + z_task (+ optional extras).
  The decoder cross-attends to this memory at every layer, so the task signal
  is equally accessible at step 1 and step T with no dilution.

API:
  forward_teacher_forcing(prev_tokens, cond_mem, mem_mask)
    → {"logits": [B, L, K], "hidden": [B, L, D]}

  forward_scheduled_sampling(prev_tokens, cond_mem, sample_prob, mem_mask)
    → {"logits": [B, L, K], "hidden": [B, L, D]}

  generate(cond_mem, sampling_cfg, mem_mask)
    → {"sequences": [B, L_out], "gen_aux": {...}, optional "hidden"}

Consumers:
  logits   → losses/ for cross-entropy reconstruction loss
  sequences → Executor (atomic action token sequence)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .sampling import Sampler


class AutoregressiveDecoder(nn.Module):
    """
    Task-conditioned autoregressive Transformer decoder.

    Conditioning: via cross-attention to external condition memory (cond_mem).

    Vocabulary: action tokens [0..K-2], EOS = K-1, PAD = -1 (never embedded).
    A learned start_emb replaces the discrete BOS token.
      Training input:  [start_emb, embed(c_1), ..., embed(c_{T-1})]
      Training target: [c_1,  c_2, ..., c_T]
    """

    def __init__(
        self,
        *,
        K_prim: int,
        d_model: int = 256,        # PDF: "隐藏维 256"
        n_layers: int = 4,         # PDF: "4 层"
        n_heads: int = 8,          # PDF: "8 头注意力"
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = -1,
        eos_id: int | None = None,
        tie_output_weights: bool = True,
    ):
        super().__init__()
        self.K_prim = int(K_prim)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id) if eos_id is not None else self.K_prim - 1
        assert 0 <= self.eos_id < self.K_prim, (
            f"eos_id={self.eos_id} out of range [0, K_prim={self.K_prim})"
        )

        # ── Token & position embeddings ────────────────────────────
        # K_prim = action_codebook_size + 1 (EOS at index K_prim-1)
        # Action tokens: 0 .. K_prim-2.  EOS: K_prim-1.  PAD: -1 (never embedded).
        self.token_embed = nn.Embedding(self.K_prim, self.d_model)
        self.pos_embed = nn.Embedding(self.max_len + 1, self.d_model)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=0.02)

        # ── Learned start embedding (replaces discrete BOS token) ──
        self.start_emb = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.start_emb, mean=0.0, std=0.02)

        # ── Core: TransformerDecoder (self-attn + cross-attn) ──────
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.dec_norm = nn.LayerNorm(self.d_model)

        # ── LM head: d_model → K_prim ─────────────────────────────
        self.lm_head = nn.Linear(self.d_model, self.K_prim, bias=False)
        if tie_output_weights:
            self.lm_head.weight = self.token_embed.weight

    # ────────────────────── internal helpers ──────────────────────

    def _causal_mask(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Upper-triangular causal mask: 0 = attend, -inf = block."""
        m = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
        return torch.triu(m, diagonal=1)

    def _positionalize(self, x: torch.Tensor) -> torch.Tensor:
        """Add learned positional embeddings to [B, L, D] tensor."""
        B, L, _ = x.shape
        if L > self.max_len + 1:
            raise ValueError(f"Sequence length {L} exceeds max_len+1={self.max_len + 1}")
        pos_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        return x + self.pos_embed(pos_ids)

    # ════════════════════════════════════════════════════════════════
    # TRAINING: teacher forcing
    # ════════════════════════════════════════════════════════════════

    def forward_teacher_forcing(
        self,
        *,
        prev_tokens: torch.Tensor,                    # [B, L-1] (left-shifted action tokens, no BOS)
        cond_mem: torch.Tensor,                        # [B, M, D] from build_cond_mem
        mem_mask: Optional[torch.Tensor] = None,       # [B, M] bool True=valid
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher-forced forward pass.

        Args:
            prev_tokens: [B, L-1] — left-shifted action tokens (no BOS; start_emb prepended internally)
            cond_mem:    [B, M, D] — condition memory from CVAE.build_cond_mem
            mem_mask:    [B, M] bool — True=valid position in memory, None=all valid

        Returns:
            logits: [B, L, K_prim] — per-position predictions over vocabulary
            hidden: [B, L, D]      — decoder hidden states
        """
        assert prev_tokens.dim() == 2
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        B, L_minus1 = prev_tokens.shape
        L = L_minus1 + 1                             # +1 for start_emb

        # embed action tokens (clamp pad_id=-1 to safe index for lookup)
        safe_tokens = prev_tokens.clamp(min=0)
        tok_emb = self.token_embed(safe_tokens)      # [B, L-1, D]

        # prepend learned start embedding
        start = self.start_emb.expand(B, 1, -1)      # [B, 1, D]
        tgt = torch.cat([start, tok_emb], dim=1)      # [B, L, D]
        tgt = self._positionalize(tgt)

        # masks
        causal_mask = self._causal_mask(L, tgt.device, tgt.dtype)
        # start_emb is always valid; pad positions in prev_tokens are ignored
        prev_pad = (prev_tokens == self.pad_id)        # [B, L-1]
        start_valid = torch.zeros(B, 1, dtype=torch.bool, device=prev_tokens.device)
        tgt_kpm = torch.cat([start_valid, prev_pad], dim=1)  # [B, L]

        # memory_key_padding_mask: True=ignored (PyTorch convention)
        mem_kpm = (~mem_mask.bool()) if mem_mask is not None else None

        # decoder: self-attn (causal) + cross-attn to cond_mem
        h = self.decoder(
            tgt=tgt,
            memory=cond_mem,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_kpm,
            memory_key_padding_mask=mem_kpm,
        )
        h = self.dec_norm(h)                         # [B, L, D]
        logits = self.lm_head(h)                     # [B, L, K]

        return {"logits": logits, "hidden": h}

    # ════════════════════════════════════════════════════════════════
    # TRAINING: scheduled sampling (bridge teacher forcing ↔ inference)
    # ════════════════════════════════════════════════════════════════

    def forward_scheduled_sampling(
        self,
        *,
        prev_tokens: torch.Tensor,                    # [B, L-1] ground-truth left-shifted
        cond_mem: torch.Tensor,                        # [B, M, D]
        sample_prob: float = 0.0,                      # p: prob of using own prediction
        mem_mask: Optional[torch.Tensor] = None,       # [B, M] bool True=valid
    ) -> Dict[str, torch.Tensor]:
        """
        Scheduled sampling: mix ground-truth and model predictions during training.

        At each decoding step t, with probability `sample_prob`, use the model's
        own argmax prediction instead of the ground-truth token as input for the
        next step. This bridges the train/inference mismatch (exposure bias).

        When sample_prob=0.0: equivalent to forward_teacher_forcing.
        When sample_prob=1.0: fully autoregressive (but still produces logits at
                              all L positions for loss computation).

        Schedule (managed externally by training loop):
          Early training:  sample_prob ≈ 0.0  →  pure teacher forcing
          Late training:   sample_prob ≈ 0.5+ →  model learns self-correction

        Args:
            prev_tokens: [B, L-1] — ground-truth left-shifted action tokens
            cond_mem:    [B, M, D] — condition memory from CVAE.build_cond_mem
            sample_prob: float in [0, 1] — probability of using own prediction
            mem_mask:    [B, M] bool — True=valid, None=all valid

        Returns:
            logits: [B, L, K_prim] — per-position predictions over vocabulary
            hidden: [B, L, D]      — decoder hidden states
        """
        # Fast path: pure teacher forcing (parallelized)
        if sample_prob <= 0.0:
            return self.forward_teacher_forcing(
                prev_tokens=prev_tokens, cond_mem=cond_mem, mem_mask=mem_mask,
            )

        assert prev_tokens.dim() == 2
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        B, L_minus1 = prev_tokens.shape
        L = L_minus1 + 1  # total output positions (start_emb + L-1 tokens)
        device = prev_tokens.device

        mem_kpm = (~mem_mask.bool()) if mem_mask is not None else None

        # Collect per-step outputs
        all_logits = []
        all_hidden = []

        # Input tokens built incrementally (may diverge from GT)
        input_tokens: list[torch.Tensor] = []   # list of [B] tensors

        for t in range(L):
            # ── build decoder input: [start_emb, embed(tok_0), ..., embed(tok_{t-1})]
            start = self.start_emb.expand(B, 1, -1)          # [B, 1, D]
            if input_tokens:
                tok_ids = torch.stack(input_tokens, dim=1)    # [B, t]
                safe_ids = tok_ids.clamp(min=0)
                tok_emb = self.token_embed(safe_ids)          # [B, t, D]
                tgt = torch.cat([start, tok_emb], dim=1)      # [B, 1+t, D]
            else:
                tgt = start                                    # [B, 1, D]

            tgt = self._positionalize(tgt)
            L_t = tgt.size(1)

            # ── masks
            causal_mask = self._causal_mask(L_t, device, tgt.dtype)

            start_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
            if input_tokens:
                tok_ids = torch.stack(input_tokens, dim=1)
                pad_mask = (tok_ids == self.pad_id)
                tgt_kpm = torch.cat([start_valid, pad_mask], dim=1)
            else:
                tgt_kpm = start_valid

            # ── forward one step through full decoder
            h = self.decoder(
                tgt=tgt,
                memory=cond_mem,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm,
            )
            h = self.dec_norm(h)
            logits_t = self.lm_head(h[:, -1, :])              # [B, K_prim]

            all_logits.append(logits_t)
            all_hidden.append(h[:, -1, :])

            # ── decide next input token (only for steps 0..L-2)
            if t < L_minus1:
                gt_token = prev_tokens[:, t]                   # [B]
                pred_token = logits_t.argmax(dim=-1)           # [B]

                # per-sample coin flip
                use_pred = torch.rand(B, device=device) < sample_prob
                next_token = torch.where(use_pred, pred_token, gt_token)
                input_tokens.append(next_token)

        logits = torch.stack(all_logits, dim=1)                # [B, L, K_prim]
        hidden = torch.stack(all_hidden, dim=1)                # [B, L, D]

        return {"logits": logits, "hidden": hidden}

    # ════════════════════════════════════════════════════════════════
    # INFERENCE: autoregressive generation via Sampler
    # ════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def generate(
        self,
        *,
        cond_mem: torch.Tensor,                        # [B, M, D]
        sampling_cfg: Dict[str, Any],
        mem_mask: Optional[torch.Tensor] = None,       # [B, M] bool True=valid
    ) -> Dict[str, Any]:
        """
        Autoregressive generation with EOS stopping and variable-length output.

        Supports all sampling strategies via Sampler:
          greedy, multinomial, top_k, nucleus, temperature, repetition_penalty.

        Args:
            cond_mem:     [B, M, D] — condition memory
            sampling_cfg: dict with strategy, temperature, max_len, min_len, etc.
            mem_mask:     [B, M] bool True=valid, None=all valid

        Returns:
            sequences: [B, L_out] — generated action token ids (may include EOS/PAD)
            gen_aux:   dict — {lengths, stopped_by_eos, max_len, min_len}
        """
        assert cond_mem.dim() == 3 and cond_mem.shape[-1] == self.d_model
        device, dtype = cond_mem.device, cond_mem.dtype
        B = cond_mem.shape[0]

        cfg = dict(sampling_cfg)

        eos_id = int(cfg.get("eos_id", self.eos_id))
        pad_id = int(cfg.get("pad_id", self.pad_id))
        max_len = int(cfg.get("max_len", self.max_len))
        min_len = int(cfg.get("min_len", 0))

        sampler = Sampler(cfg)

        # generated action tokens (initially empty — start_emb used instead of BOS)
        generated = torch.zeros(B, 0, dtype=torch.long, device=device)
        alive = torch.ones(B, dtype=torch.bool, device=device)
        stopped_by_eos = torch.zeros(B, dtype=torch.bool, device=device)
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        # memory padding mask (invert: True=valid → True=ignored for PyTorch)
        mem_kpm = (~mem_mask.bool()) if mem_mask is not None else None

        for t in range(max_len):
            # build decoder input: [start_emb, embed(c_0), ..., embed(c_{t-1})]
            start = self.start_emb.expand(B, 1, -1)            # [B, 1, D]
            if generated.size(1) > 0:
                safe_gen = generated.clamp(min=0)
                tok_emb = self.token_embed(safe_gen)            # [B, t, D]
                tgt = torch.cat([start, tok_emb], dim=1)        # [B, 1+t, D]
            else:
                tgt = start                                      # [B, 1, D]
            tgt = self._positionalize(tgt)
            L_t = tgt.size(1)

            causal_mask = self._causal_mask(L_t, device, dtype)

            # pad mask: start_emb always valid, pad positions ignored
            start_valid = torch.zeros(B, 1, dtype=torch.bool, device=device)
            if generated.size(1) > 0:
                gen_pad = (generated == pad_id)
                tgt_kpm = torch.cat([start_valid, gen_pad], dim=1)
            else:
                tgt_kpm = start_valid

            h = self.decoder(
                tgt=tgt,
                memory=cond_mem,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_kpm,
                memory_key_padding_mask=mem_kpm,
            )
            h = self.dec_norm(h)                       # [B, L_t, D]
            logits_t = self.lm_head(h[:, -1, :])       # [B, K]

            # sample next token
            pick = sampler.step(
                logits=logits_t,
                step=t,
                generated_ids=generated if generated.size(1) > 0 else None,
            )
            next_ids = pick["ids"]                     # [B]

            # enforce min_len: block EOS before min_len steps
            if t < min_len:
                is_eos = next_ids.eq(eos_id)
                if is_eos.any():
                    fallback_logits = pick.get("filtered_logits", logits_t).clone()
                    fallback_logits[:, eos_id] = float("-inf")
                    next_ids = torch.where(
                        is_eos,
                        torch.argmax(fallback_logits, dim=-1),
                        next_ids,
                    )

            # pad already-stopped sequences
            next_ids = torch.where(alive, next_ids, torch.full_like(next_ids, pad_id))

            # append to generated sequence
            generated = torch.cat([generated, next_ids.unsqueeze(-1)], dim=1)

            # check EOS
            just_eos = (next_ids == eos_id) & (t + 1 >= min_len)
            newly_finished = alive & just_eos

            # Record the exact finishing step before stopped_by_eos flips.
            lengths = torch.where(
                newly_finished,
                torch.full_like(lengths, t + 1),
                lengths,
            )
            stopped_by_eos |= newly_finished
            alive = alive & (~just_eos)

            if not alive.any():
                break

        # For sequences that never emitted EOS, length is the generated span.
        lengths = torch.where(
            stopped_by_eos,
            lengths,
            torch.full_like(lengths, generated.size(1)),
        )

        sequences = generated                          # action tokens only, no BOS to strip

        gen_aux = {
            "lengths": lengths,
            "stopped_by_eos": stopped_by_eos,
            "max_len": max_len,
            "min_len": min_len,
        }

        out: Dict[str, Any] = {"sequences": sequences, "gen_aux": gen_aux}

        return out
