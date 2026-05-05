"""
LanguageEncoder — Text → Task Token Interface (Path B)

Pipeline (Stage 4.1, Path B):
  "open the door" → frozen text encoder → v_raw [B, D_text]
                  → learned projection   → text_emb [B, task_dim]
                  → NN lookup (L2)        → task_id [B], task_emb [B, task_dim]

TRAINING:  encode(texts) → text_emb [B, task_dim]
           → fed to losses/info_nce_loss.py for InfoNCE(task_emb, text_emb)

INFERENCE: infer_task_token(texts, task_codebook) → task_id, task_emb, text_emb
           → task_id, task_emb, fed to CVAE (Stage 4.2) → Decoder (Stage 4.3)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer

# ══════════════════════════════════════════════════════════════════
# Internal Text Encoder (frozen pretrained backbone)
# ══════════════════════════════════════════════════════════════════
class _TextEncoder(nn.Module):
    """
    Frozen pretrained text encoder: raw strings → embeddings.

    Supports two backends:
      - "clip": HuggingFace CLIP (openai/clip-vit-large-patch14, 768-dim)
      - "sentencetransformer": all-mpnet-base-v2 (768-dim)

    Output: v_raw [B, embedding_dim]
    """

    def __init__(self, type: str):
        super().__init__()
        self.enc_type = type.lower()

        if self.enc_type == "sentencetransformer":
            self.pretrained_name = "all-mpnet-base-v2"
            self.model = SentenceTransformer(self.pretrained_name)
            self.out_dim = int(self.model.get_sentence_embedding_dimension())
            self._use_clip_model = False

        elif self.enc_type == "clip":
            self.pretrained_name = "openai/clip-vit-large-patch14"
            self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_name)
            try:
                self.model = CLIPModel.from_pretrained(self.pretrained_name)
                self.out_dim = int(self.model.config.projection_dim)
                self._use_clip_model = True
            except Exception:
                self.model = CLIPTextModel.from_pretrained(self.pretrained_name)
                self.out_dim = int(self.model.config.hidden_size)
                self._use_clip_model = False

        else:
            raise ValueError(f"Unknown text encoder type: {self.enc_type}")

        # Freeze all parameters
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _default_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def forward(
        self, texts: List[str], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Args:
            texts: List of B non-empty strings.
            device: Target device. If None, inferred from model parameters.

        Returns:
            v_raw: [B, embedding_dim] float32, detached (no grad through frozen encoder)
        """
        if not texts:
            raise ValueError("texts is empty.")
        if any(t is None or len(str(t).strip()) == 0 for t in texts):
            raise ValueError("texts contains empty or None string.")

        if device is None:
            device = self._default_device()

        with torch.no_grad():
            if self.enc_type == "clip":
                tokens = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}

                if self._use_clip_model:
                    v = self.model.get_text_features(**tokens)
                else:
                    outputs = self.model(**tokens)
                    last_hidden = outputs.last_hidden_state
                    attn_mask = tokens["attention_mask"]
                    eos_idx = attn_mask.sum(dim=1) - 1
                    batch_idx = torch.arange(last_hidden.size(0), device=device)
                    v = last_hidden[batch_idx, eos_idx]

            else:  # sentencetransformer
                v = self.model.encode(
                    texts,
                    batch_size=len(texts),
                    convert_to_tensor=True,
                    normalize_embeddings=False,
                    show_progress_bar=False,
                )
                if v.dtype != torch.float32:
                    v = v.float()
                v = v.to(device, non_blocking=True)

        # sentence-transformers .encode() uses @torch.inference_mode()
        # internally, producing inference-mode tensors that cannot be
        # saved for backward.  .detach().clone() outside the no_grad
        # block yields a normal leaf tensor safe for autograd.
        return v.detach().clone()

    def train(self, mode: bool = True):
        """Always stay in eval mode — frozen encoder."""
        super().train(mode)
        self.model.eval()
        return self


# ══════════════════════════════════════════════════════════════════
# Projection Head (trainable)
# ══════════════════════════════════════════════════════════════════
class _ProjectionHead(nn.Module):
    """
    Learned projection: v_raw ∈ R^embedding_dim → v_proj ∈ R^task_dim.
    """

    def __init__(
        self,
        proj_type: str = "mlp",
        in_dim: int = 768,
        task_dim: int = 64,
        hidden: int = 512,
        act: str = "gelu",
        normalize: bool = False,
    ):
        super().__init__()
        proj_type = str(proj_type).lower()
        self.in_dim = int(in_dim)
        self.task_dim = int(task_dim)
        self.normalize = bool(normalize)

        if proj_type == "linear":
            self.proj = nn.Linear(in_dim, task_dim)

        elif proj_type == "mlp":
            act_layer = {"gelu": nn.GELU(), "relu": nn.ReLU()}.get(
                str(act).lower(), nn.Identity()
            )
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden),
                act_layer,
                nn.Linear(hidden, task_dim),
            )
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")

        self._init_weights()

    def _init_weights(self):
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, v_raw: torch.Tensor) -> torch.Tensor:
        v_proj = self.proj(v_raw)
        if self.normalize:
            v_proj = F.normalize(v_proj, dim=-1, eps=1e-8)
        return v_proj


# ══════════════════════════════════════════════════════════════════
# Language Encoder
# ══════════════════════════════════════════════════════════════════
class LanguageEncoder(nn.Module):
    """
    Text → task codebook space.

    TRAINING:  encode(texts) → v_proj [B, task_dim]
    INFERENCE: infer_task_token(texts, task_codebook) → task_id, u_j, v_proj
    """

    def __init__(
        self,
        model_type: str,
        proj_cfg: Dict,
    ):
        super().__init__()

        # ── Frozen text encoder ───────────────────────────────────
        self.text_enc = _TextEncoder(model_type)

        # ── Trainable projection head ─────────────────────────────
        proj_cfg = dict(proj_cfg) 
        proj_cfg["in_dim"] = self.text_enc.out_dim
        self.proj_head = _ProjectionHead(**proj_cfg)

    # ══════════════════════════════════════════════════════════════
    # INTERNAL: encoding pipeline
    # ══════════════════════════════════════════════════════════════

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Raw strings → frozen text embeddings.

        Returns:
            v_raw: [B, embedding_dim] — no gradient (frozen encoder).
        """
        device = next(self.proj_head.parameters()).device
        return self.text_enc(texts, device=device)

    def _project(self, v_raw: torch.Tensor) -> torch.Tensor:
        """
        Frozen embeddings → task-space embeddings.

        Returns:
            v_proj: [B, task_dim] — has gradient (trainable projection).
        """
        return self.proj_head(v_raw)

    # ══════════════════════════════════════════════════════════════
    # INTERNAL: L2 matching (consistent with TaskTokenizer's VQ)
    # ══════════════════════════════════════════════════════════════
    def _match(
        self,
        v_proj: torch.Tensor,        # [B, task_dim]
        task_codebook: torch.Tensor,  # [J, task_dim]
    ) -> torch.LongTensor:
        """
        Nearest-neighbor lookup via L2 distance.

        Returns:
            c_task: [B] long — index of nearest codebook entry.
        """
        # L2 distance: ||v - u||^2 = ||v||^2 - 2*v·u + ||u||^2
        dists = (
            v_proj.pow(2).sum(dim=-1, keepdim=True)
            - 2 * v_proj @ task_codebook.t()
            + task_codebook.pow(2).sum(dim=-1, keepdim=True).t()
        )
        return dists.argmin(dim=-1).long()

    # ══════════════════════════════════════════════════════════════
    # TRAINING
    # ══════════════════════════════════════════════════════════════

    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        TRAINING: raw strings → projected embeddings in task codebook space.

        Only returns v_proj — the InfoNCE loss in losses/ pairs this with u_j from TaskTokenizer. No codebook lookup needed here.

        Returns:
            v_proj: [B, task_dim] — WITH gradient through projection head.
        """
        text_raw = self._encode_text(texts)   # [B, embedding_dim], text to embedding
        text_emb = self._project(text_raw)      # [B, task_dim], embedding to task space
        return text_emb

    # ══════════════════════════════════════════════════════════════
    # INFERENCE
    # ══════════════════════════════════════════════════════════════

    @torch.no_grad()
    def infer_task_token(
        self,
        texts: List[str],
        task_codebook: torch.Tensor,  # [J, task_dim]
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        INFERENCE: raw strings → task token via nearest codebook entry.

        Args:
            texts: List of B strings.
            task_codebook: [J, task_dim] — from TaskTokenizer.codebook.weight

        Returns:
            task_id:         [B] long        — discrete task token index
            task_emb:     [B, task_dim]   — codebook[task_id]
            text_emb:     [B, task_dim]   — projected text embedding (pre-quantization)
        """
        text_emb = self.encode(texts)                          # [B, task_dim]    text to task space
        task_id = self._match(text_emb, task_codebook)          # [B]              Nearest-neighbor lookup via L2 distance. Get taskId in task_codebook
        task_emb = F.embedding(task_id, task_codebook)             # [B, task_dim]    
        return task_id, task_emb, text_emb

    # ══════════════════════════════════════════════════════════════
    # Training mode override
    # ══════════════════════════════════════════════════════════════

    def train(self, mode: bool = True):
        """Keep frozen encoder in eval mode; only projection head trains."""
        super().train(mode)
        self.text_enc.model.eval()
        return self