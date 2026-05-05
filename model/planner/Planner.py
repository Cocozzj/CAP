"""
Planner — Orchestrator for Conditional Probabilistic Planning (Stage 4)

Role:
  Thin orchestrator. Owns three submodules, calls them in order,
  passes outputs between them. Has ZERO neural layers of its own.

Two modes:
  Training  (seq_tokens given):  TaskTokenizer → posterior → cond_mem → teacher forcing
  Inference (task_emb or text):  prior → cond_mem → autoregressive generate

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .functions import LanguageEncoder,TaskTokenizer
from .CVAE import CVAE


class Planner(nn.Module):
    def __init__(
        self,
        *,
        specials: Dict[str, int],
        sampling_cfg: Dict[str, Any],
        cvae_cfg: Dict[str, Any],
        language_encoder_cfg: Dict[str, Any],
        task_tokenizer_cfg: Dict[str, Any],
        use_task_token: bool = True,
    ):
        """
        Args:
            specials:  {"eos_id": 512, "pad_id": -1}
                       pad_id also used by TaskTokenizer._build_padding_mask
            sampling_cfg:  default inference config
                {"strategy": "top_k", "top_k": 50, "temperature": 0.8,
                 "max_len": 30, "num_samples": 1, "deterministic": True}
            cvae_cfg:             kwargs for CVAE.__init__
            language_encoder_cfg: kwargs for LanguageEncoder.__init__
            task_tokenizer_cfg:   kwargs for TaskTokenizer.__init__
            use_task_token:       PDF main proposal §5.2 ablation table
                "w/o hierarchical" — when False, the TaskTokenizer layer is
                bypassed entirely.  text → LanguageEncoder → text_emb
                directly conditions the CVAE / AR decoder, skipping the
                discrete task codebook.  Atomic codebook is unaffected.
        """
        super().__init__()
        self.use_task_token = bool(use_task_token)

        # ── submodules ──────────────────────────────────────────────
        # TaskTokenizer is constructed only when hierarchical mode is on
        self.task_tok = TaskTokenizer(**task_tokenizer_cfg) if self.use_task_token else None
        self.cvae     = CVAE(**cvae_cfg)
        self.lang     = LanguageEncoder(**language_encoder_cfg)

        # ── special token IDs ───────────────────────────────────────
        self._pad = int(specials.get("pad_id", -1))
        self._eos = int(specials.get("eos_id", 512))

        # ── default inference config ────────────────────────────────
        self.sampling_cfg  = dict(sampling_cfg)
        self.num_samples   = int(sampling_cfg.get("num_samples", 1))
        self.deterministic = bool(sampling_cfg.get("deterministic", True))

    # ================================================================
    # Read-only accessors
    # ================================================================

    def task_codebook_weight(self) -> Optional[torch.Tensor]:
        """Task VQ codebook weight tensor [J, task_dim]; None if hierarchical disabled."""
        if self.task_tok is None:
            return None
        return self.task_tok.codebook.weight.detach()

    # ================================================================
    # Training: TaskTokenizer → posterior → cond_mem → teacher forcing
    # ================================================================

    def training_forward(
        self,
        *,
        token_indices:      torch.Tensor,            # [B, L] long
        atomic_codebook: torch.Tensor,              # [K, atomic_dim]
        text_labels:     Optional[List[str]] = None,
        sample_prob:     float = 0.0,                # scheduled sampling probability
        deterministic:   bool = False,               # No-CVAE ablation: z = mu (no sampling)
    ) -> Dict[str, Any]:

        B, L = token_indices.shape
        device = token_indices.device

        safe_indices = token_indices.clamp(0, atomic_codebook.size(0) - 1)
        token_embeds = F.embedding(safe_indices, atomic_codebook)    #[B,L,atomic_dim]

        # ── Step 1: Obtain task_emb ─────────────────────────────────
        # Hierarchical (default): TaskTokenizer compresses atomic seq → task_emb
        # Ablation (use_task_token=False): use language_encoder(text_labels) instead
        if self.use_task_token:
            task_id, task_emb, h_task, vq_loss = self.task_tok.encode(
                token_indices=token_indices,       # original (with pad=-1) for padding mask
                token_embeds=token_embeds,          # safe embeddings from clamped indices
            )
            # task_id:   [B] long        — discrete task token index
            # task_emb:  [B, task_dim]   — quantized (straight-through)
            # h_task:    [B, task_dim]   — pre-VQ continuous
            # vq_loss:   scalar
        else:
            if text_labels is None:
                raise ValueError(
                    "Planner(use_task_token=False) requires text_labels at training time "
                    "(text → language_encoder → task_emb path is the only conditioning source)."
                )
            text_emb = self.lang.encode(text_labels)            # [B, task_dim]
            task_emb = text_emb
            h_task   = text_emb                                  # for L_NCE compatibility
            vq_loss  = task_emb.new_zeros(())
            task_id  = torch.zeros(B, dtype=torch.long, device=device)

        # ── Step 2: Reconstruction (for external task_recon loss) ───
        # Only meaningful with TaskTokenizer; in ablation just echo task_emb.
        recon_h_task = (self.task_tok.reconstruct(task_emb)
                        if self.use_task_token else task_emb)
        
        # ── Step 3: CVAE posterior q_φ(z | seq, task) ───────────────
        #   Reads GT sequence + task → infers latent z capturing style
        post = self.cvae.encode_posterior(
            token_indices=token_indices,
            task_emb=task_emb,
            seq_mask=(token_indices != self._pad),      # True = valid
        )
        mu, logvar, z = post["mu"], post["logvar"], post["z"]
        if deterministic:
            z = mu  # No-CVAE ablation: skip reparameterization, use mean directly

        # ── Step 4: Build condition memory ──────────────────────────
        #   (task_emb, z) → cond_mem [B, M, D] for cross-attention
        cond_mem, mem_mask = self.cvae.build_cond_mem(
            task_emb=task_emb,
            z_task=z,
        )

        # ── Step 5: Build targets with EOS ─────────────────────────────
        #   Place EOS after the last valid (non-pad) token so the AR
        #   decoder learns when to stop.
        #   targets:      [c1, ..., c_T, EOS, pad, ..., pad]   shape [B, L]
        #   prev_tokens:  [c1, ..., c_{L-1}]                   shape [B, L-1]
        #   Decoder prepends start_emb → input length L → logits [B, L, K]
        #
        #   If the sequence is fully packed (seq_lengths == L), we cannot
        #   place EOS without overwriting the last real token.  In that case
        #   we leave the targets as-is (no EOS), so the decoder still learns
        #   all action tokens. The loss mask should ignore the missing EOS.
        targets = token_indices.clone()
        valid_mask = (token_indices != self._pad)
        seq_lengths = valid_mask.sum(dim=1)                         # [B]
        can_place_eos = seq_lengths < L                              # [B] bool
        eos_pos = seq_lengths.clamp(max=L - 1)                      # [B]
        batch_idx = torch.arange(B, device=device)
        targets[batch_idx[can_place_eos], eos_pos[can_place_eos]] = self._eos

        prev_tokens = token_indices[:, :-1]

        # ── Step 6: Decoding (teacher forcing or scheduled sampling) ──
        if sample_prob > 0.0:
            dec_out = self.cvae.decode_scheduled_sampling(
                token_indices=prev_tokens,
                cond_mem=cond_mem,
                sample_prob=sample_prob,
                mem_mask=mem_mask,
            )
        else:
            dec_out = self.cvae.decode_teacher_forcing(
                token_indices=prev_tokens,
                cond_mem=cond_mem,
                mem_mask=mem_mask,
            )

        # ── Step 7: Text projection for InfoNCE (optional) ─────────
        v_proj = None
        if text_labels is not None:
            v_proj = self.lang.encode(text_labels)   # [B, task_dim]

        return {
            "logits":       dec_out["logits"],       # [B, L, K_prim]
            "targets":      targets,                 # [B, L] with EOS after last valid token
            "mu":           mu,                      # [B, z_dim]
            "logvar":       logvar,                  # [B, z_dim]
            "z":            z,                       # [B, z_dim]
            "vq_loss":      vq_loss,                 # scalar
            "task_emb":     task_emb,                # [B, task_dim]
            "task_id":      task_id,                 # [B] long
            "h_task":       h_task,                  # [B, task_dim]
            "recon_h_task": recon_h_task,             # [B, task_dim]
            "v_proj":       v_proj,                  # [B, task_dim] or None
        }

    # ================================================================
    # Inference: prior → cond_mem → autoregressive generate
    # ================================================================
    @torch.no_grad()
    def sample_actions(
        self,
        *,
        texts: List[str],
        sampling_info: Optional[Dict] = None,
        num_samples:   Optional[int]  = None,
    ) -> Dict[str, Any]:
        """
        Data flow:
          texts → infer_task_from_text → task_emb
          task_emb → repeat(N) → [B*N, task_dim]
          z ~ N(0,I)           → [B*N, z_dim]
          build_cond_mem       → cond_mem [B*N, M, D]
          AR generate loop     → sequences [B*N, L_out]

        Returns:
            sequences  [B*N, L_out]  — generated atomic token ids
            gen_aux    dict          — {lengths, stopped_by_eos, ...}
        """
        cfg = dict(self.sampling_cfg)
        if sampling_info is not None:
            cfg.update(sampling_info)

        N = int(num_samples or cfg.get("num_samples", self.num_samples))
        deterministic = bool(cfg.get("deterministic", self.deterministic))
        
        inf = self.infer_task_from_text(texts)
        task_emb = inf["task_emb"]

        B = task_emb.size(0)
        device = task_emb.device
        dtype  = task_emb.dtype

        task_rep = task_emb.repeat_interleave(N, dim=0)       # [B*N, task_dim]

        if deterministic:
            z = torch.zeros(B * N, self.cvae.z_dim, device=device, dtype=dtype)
        else:
            z = self.cvae.sample_prior(B * N, device=device)  # [B*N, z_dim]

        cond_mem, mem_mask = self.cvae.build_cond_mem(
            task_emb=task_rep,
            z_task=z,
        )

        gen_out = self.cvae.decode_generate(
            cond_mem=cond_mem,
            sampling_cfg=cfg,
            mem_mask=mem_mask,
        )
        gen_out["task_emb"] = task_emb                           # [B, task_dim]
        return gen_out


    # ================================================================
    # Utility: composite task planning
    # ================================================================
    @torch.no_grad()
    def plan_composite(
        self,
        task_embs: List[torch.Tensor],            # list of [B, task_dim]
        sampling_info: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Plan a composite task: generate sub-sequences, concatenate.

        PDF §4.1: "复合任务 c_task^1 ∘ c_task^2 → seq_1 ++ seq_2"

        Returns:
            full_seq  [B, sum(L_i)]  — concatenated plan
            sub_seqs  list of [B, L_i]
        """
        cfg = dict(self.sampling_cfg)
        if sampling_info is not None:
            cfg.update(sampling_info)

        deterministic = bool(cfg.get("deterministic", self.deterministic))

        sub_seqs = []
        for t_emb in task_embs:
            B = t_emb.size(0)
            device = t_emb.device
            dtype  = t_emb.dtype

            if deterministic:
                z = torch.zeros(B, self.cvae.z_dim, device=device, dtype=dtype)
            else:
                z = self.cvae.sample_prior(B, device=device)

            cond_mem, mem_mask = self.cvae.build_cond_mem(
                task_emb=t_emb,
                z_task=z,
            )
            out = self.cvae.decode_generate(
                cond_mem=cond_mem,
                sampling_cfg=cfg,
                mem_mask=mem_mask,
            )
            sub_seqs.append(out["sequences"])

        return {
            "full_seq": torch.cat(sub_seqs, dim=1),
            "sub_seqs": sub_seqs,
        }

    # ================================================================
    # Utility: text → task token
    # ================================================================
    @torch.no_grad()
    def infer_task_from_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Text → task embedding.

        Hierarchical (default): nearest-neighbour lookup in the task codebook.
        Ablation (use_task_token=False): the projected text embedding IS the
        task embedding — no quantisation, no codebook lookup.

        Returns:
            task_id   [B] long       — codebook index (None / zeros in ablation)
            task_emb  [B, task_dim]  — codebook[task_id]  OR  text_emb (ablation)
            text_emb  [B, task_dim]  — projected text embedding (pre-quantization)
        """
        if self.use_task_token:
            cb_weight = self.task_codebook_weight()           # [J, task_dim]
            task_id, task_emb, text_emb = self.lang.infer_task_token(texts, cb_weight)
        else:
            text_emb = self.lang.encode(texts)                # [B, task_dim]
            task_emb = text_emb
            B = text_emb.size(0)
            task_id = torch.zeros(B, dtype=torch.long, device=text_emb.device)
        return {"task_id": task_id, "task_emb": task_emb, "text_emb": text_emb}

    # ================================================================
    # Train mode override
    # ================================================================

    def train(self, mode: bool = True):
        """Keep frozen text encoder in eval regardless of training mode.

        Note: LanguageEncoder.train() already keeps text_enc in eval mode,
        so no extra handling needed here beyond the standard super().train().
        """
        super().train(mode)
        return self