# -*- coding: utf-8 -*-
"""
Token sampling utilities for autoregressive decoding.

Strategies:
  - greedy                 : argmax over logits
  - multinomial            : categorical sampling with temperature
  - top_k                  : keep top-k probs then sample
  - nucleus (top_p)        : keep smallest set with cumulative prob >= p then sample

Extras:
  - temperature            : diversity knob (>0), applied before filtering
  - repetition_penalty     : punish tokens that already appeared in the sequence
  - min_len & eos gating   : forbid EOS before min_len
  - pad suppression        : never sample PAD; ignore PAD in repetition stats
  - deterministic seed     : reproducible stochastic sampling
"""

from __future__ import annotations
from typing import Optional, Tuple, Sequence, Dict, Any

import torch
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════

def _apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature is None or temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        return logits  # treat as greedy
    return logits / float(temperature)


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply 'CTRL' style repetition penalty per batch row.
    If token was generated before:
      - if logit > 0: divide by penalty
      - if logit < 0: multiply by penalty
    logits      : [B, V]
    generated_ids: [B, L] int64
    """
    if penalty is None or penalty <= 1.0 or generated_ids is None or generated_ids.numel() == 0:
        return logits

    penalized = logits.clone()
    B, V = penalized.shape

    # Build per-batch mask of seen tokens, ignoring negatives and PAD
    seen = torch.zeros_like(penalized, dtype=torch.bool)  # [B, V]
    for b in range(B):
        ids = generated_ids[b]
        if ids.numel() == 0:
            continue
        mask = ids >= 0
        if pad_id is not None:
            mask &= (ids != pad_id)
        ids = ids[mask]
        if ids.numel() > 0:
            ids = torch.unique(ids.clamp_(0, V - 1))
            if ids.numel() > 0:
                seen[b, ids] = True

    positive = penalized > 0
    penalized = torch.where(seen & positive, penalized / penalty, penalized)
    penalized = torch.where(seen & (~positive), penalized * penalty, penalized)
    return penalized


def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keep only top_k tokens (per row), set others to -inf."""
    if top_k is None or int(top_k) <= 0:
        return logits
    top_k = min(int(top_k), logits.shape[-1])
    values, _ = torch.topk(logits, top_k, dim=-1)
    cutoffs = values[..., -1, None]  # [B, 1]
    return torch.where(logits < cutoffs, torch.finfo(logits.dtype).min, logits)


def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus (top-p) filtering: keep the smallest set with cumulative prob >= p.
    """
    if top_p is None:
        return logits
    top_p = float(top_p)
    if top_p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumprobs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_mask = cumprobs > top_p
    cutoff_mask[..., 0] = False  # always keep at least one

    mask_orig = torch.zeros_like(logits, dtype=torch.bool)
    mask_orig.scatter_(dim=-1, index=sorted_idx, src=cutoff_mask)

    filtered = logits.clone()
    filtered[mask_orig] = torch.finfo(logits.dtype).min
    return filtered


def _sample_categorical(
    logits: torch.Tensor,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multinomial sample per row.
    Returns:
      ids:   Long[B]
      probs: Float[B]
    """
    gen = None
    if seed is not None:
        gen = torch.Generator(device=logits.device)
        gen.manual_seed(int(seed))

    probs = F.softmax(logits, dim=-1)
    ids = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
    picked = probs.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
    return ids.long(), picked


# ══════════════════════════════════════════════════════════════════
# Core entry point (functional)
# ══════════════════════════════════════════════════════════════════

def pick_tokens(
    logits: torch.Tensor,
    *,
    strategy: str = "greedy",       # "greedy" | "multinomial" | "top_k" | "nucleus"
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    generated_ids: Optional[torch.Tensor] = None,  # [B, L_so_far]
    eos_id: Optional[int] = None,
    pad_id: Optional[int] = None,
    step: int = 0,
    min_len: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Core entry point: pick next token ids given logits.

    Args:
      logits: [B, V] float logits
      strategy: "greedy" | "multinomial" | "top_k" | "nucleus"
      generated_ids: already generated prefix for repetition penalty
      eos_id/min_len: gating EOS before min_len
      pad_id: permanently banned from sampling
      seed: sampling RNG seed (per-step)

    Returns:
      {
        "ids": Long[B],
        "probs": Float[B],         # prob of the chosen token
        "filtered_logits": Float[B, V]  # after temperature + filters + gating
      }
    """
    assert logits.dim() == 2, "logits must be [B, V]"
    B, V = logits.shape

    # dtype-normalize
    x = logits.to(dtype=torch.float32)

    # NaN/Inf guard — torch.multinomial / softmax produce CUDA device-side
    # asserts when logits contain non-finite values.  Replace NaN/+Inf with
    # a large-but-finite floor; -Inf stays as a hard mask (unreachable).
    # Without this guard, transient training instabilities (early CVAE/AR
    # before convergence) crash the whole run instead of just producing
    # a low-quality sample.
    finfo_max = torch.finfo(x.dtype).max
    x = torch.nan_to_num(x, nan=0.0, posinf=finfo_max / 2, neginf=-finfo_max / 2)

    # repetition penalty (ignore PAD)
    if generated_ids is not None:
        x = _apply_repetition_penalty(x, generated_ids, float(repetition_penalty), pad_id=pad_id)

    # temperature
    x = _apply_temperature(x, float(temperature))

    # gating: forbid PAD and (before min_len) EOS
    inf_neg = torch.finfo(x.dtype).min
    if pad_id is not None and 0 <= int(pad_id) < V:
        x[:, int(pad_id)] = inf_neg
    if eos_id is not None and step < int(min_len) and 0 <= int(eos_id) < V:
        x[:, int(eos_id)] = inf_neg

    # strategy-specific filtering + pick
    strat = str(strategy).lower()
    if strat == "greedy" or (temperature is not None and float(temperature) <= 0):
        ids = torch.argmax(x, dim=-1)
        probs = F.softmax(x, dim=-1).gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return {"ids": ids.long(), "probs": probs, "filtered_logits": x}

    elif strat == "multinomial":
        ids, picked = _sample_categorical(x, seed=seed)
        return {"ids": ids, "probs": picked, "filtered_logits": x}

    elif strat == "top_k":
        xk = _top_k_filtering(x, int(top_k))
        ids, picked = _sample_categorical(xk, seed=seed)
        return {"ids": ids, "probs": picked, "filtered_logits": xk}

    elif strat == "nucleus":
        xp = _top_p_filtering(x, float(top_p))
        ids, picked = _sample_categorical(xp, seed=seed)
        return {"ids": ids, "probs": picked, "filtered_logits": xp}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ══════════════════════════════════════════════════════════════════
# Sampler class (stateful config wrapper)
# ══════════════════════════════════════════════════════════════════

class Sampler:
    """
    Thin stateless wrapper to avoid passing a dozen kwargs every time.
    Accepts either a dict-like cfg or an object with attributes (e.g., dataclass).
    """
    def __init__(self, sample_cfg: Any):
        self.cfg = sample_cfg

    def _get(self, key: str, default: Any = None) -> Any:
        # Support dict or attr-based config
        if isinstance(self.cfg, dict):
            return self.cfg.get(key, default)
        return getattr(self.cfg, key, default)

    def step(
        self,
        logits: torch.Tensor,
        *,
        step: int = 0,
        generated_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return pick_tokens(
            logits,
            strategy=str(self._get("strategy", "multinomial")),
            temperature=float(self._get("temperature", 1.0)),
            top_k=int(self._get("top_k", 0)),
            top_p=float(self._get("top_p", 1.0)),
            repetition_penalty=float(self._get("repetition_penalty", 1.0)),
            generated_ids=generated_ids,
            eos_id=(int(self._get("eos_id")) if self._get("eos_id") is not None else None),
            pad_id=(int(self._get("pad_id")) if self._get("pad_id") is not None else None),
            step=int(step),
            min_len=int(self._get("min_len", 0)),
            seed=(int(self._get("seed")) if self._get("seed") is not None else None),
        )