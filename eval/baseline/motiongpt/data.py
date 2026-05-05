"""Dataset adapter for MotionGPT-style fine-tuning on our pose-delta data.

MotionGPT treats motion as a "foreign language" — text and motion live in
the same vocabulary, separated by special tokens.  We adapt the same idea
on our smaller dataset:

  Input  text:   "open the box"
  Target motion: <motion_start> <m_42> <m_17> ... <m_88> <motion_end>

where <m_i> are motion token IDs in [0, K), produced by VQ-encoding the
per-frame pose deltas.  We reuse the FlatVQVAE codebook here so the
comparison shares the same motion vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..flat_vqvae.data import FlatVQVAEDataset, pose_to_delta


# ──────────────────────────────────────────────────────────────────────
# Special tokens
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MGSpecialTokens:
    """Convention for our MotionGPT vocab extension.

    The base T5 tokenizer's vocab has ~32k entries.  We append:
      <motion_start>           ID = 32000
      <motion_end>             ID = 32001
      <m_0>, <m_1>, ..., <m_K-1>  IDs = 32002 .. 32002+K-1
    """
    motion_start: str = "<motion_start>"
    motion_end:   str = "<motion_end>"
    motion_token: str = "<m_{}>"           # format string

    @classmethod
    def all_special_tokens(cls, K: int) -> List[str]:
        return [cls.motion_start, cls.motion_end] + [
            cls.motion_token.format(i) for i in range(K)
        ]


# ──────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MGSample:
    text:        str
    deltas:      np.ndarray         # [T-1, 7]  per-frame pose delta (training target before tokenization)
    motion_ids:  Optional[np.ndarray] = None   # [T-1] int (filled by training loop after VQ encode)
    pose0:       Optional[np.ndarray] = None
    traj_id:     str = ""


class MotionGPTDataset(Dataset):
    """Wraps FlatVQVAEDataset; identical I/O semantics."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir:      Union[str, Path],
        split:         str = "train",
        T:             int = 30,
    ):
        self._inner = FlatVQVAEDataset(manifest_path, data_dir, split=split, T=T)

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, i: int) -> MGSample:
        s = self._inner[i]
        return MGSample(
            text=s.text, deltas=s.deltas, pose0=s.pose0, traj_id=s.traj_id,
        )


def collate_mg(samples: Sequence[MGSample]) -> Dict:
    deltas = np.stack([s.deltas for s in samples], axis=0)            # [B, T-1, 7]
    pose0  = np.stack([s.pose0  for s in samples], axis=0) if samples[0].pose0 is not None else None
    return {
        "deltas":  torch.from_numpy(deltas).float(),
        "pose0":   torch.from_numpy(pose0).float() if pose0 is not None else None,
        "texts":   [s.text for s in samples],
        "traj_id": [s.traj_id for s in samples],
    }


# ──────────────────────────────────────────────────────────────────────
# Format text + motion-ids into the T5-input string
# ──────────────────────────────────────────────────────────────────────

def format_input_text(text: str) -> str:
    """T5 source: just the natural-language instruction (no special tokens)."""
    return text


def format_target_motion(motion_ids: Sequence[int],
                          specials: MGSpecialTokens = MGSpecialTokens(),
                          ) -> str:
    """T5 target: <motion_start> <m_42> <m_17> ... <motion_end>"""
    body = " ".join(specials.motion_token.format(int(i)) for i in motion_ids)
    return f"{specials.motion_start} {body} {specials.motion_end}"


def parse_motion_ids_from_text(generated: str,
                                 specials: MGSpecialTokens = MGSpecialTokens(),
                                 ) -> List[int]:
    """Inverse of ``format_target_motion`` — extract integer IDs."""
    import re
    pat = re.compile(r"<m_(\d+)>")
    return [int(m.group(1)) for m in pat.finditer(generated)]
