"""Dataset adapter for MotionGPT-style fine-tuning on our pose-delta data.

MotionGPT treats motion as a "foreign language" — text and motion live in
the same vocabulary, separated by special tokens.  We adapt the same idea
on our smaller dataset:

  Input  text:   "open the box"
  Target motion: <motion_start> <m_42> <m_17> ... <m_88> <motion_end>

where <m_i> are motion token IDs in [0, K), produced by VQ-encoding the
per-frame pose deltas with our self-contained ``vqvae.FlatVQVAE``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from ..common import iter_split_entries
from dataload.text import task_to_text


# ──────────────────────────────────────────────────────────────────────
# Pose-delta math (xyzw quaternion convention, same as motion.py / runner)
# ──────────────────────────────────────────────────────────────────────

def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.maximum(n, 1e-12)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xyzw quaternion product, broadcasted on the last axis."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return np.stack([x, y, z, w], axis=-1)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse for unit quaternion = conjugate (negate xyz, keep w)."""
    out = q.copy()
    out[..., :3] = -out[..., :3]
    return out


def pose_to_delta(poses: np.ndarray) -> np.ndarray:
    """Convert absolute pose trajectory [T, 7] (xyz + quat-xyzw) to per-step
    deltas [T-1, 7] in the previous pose's local frame.

    Translation delta: p_{t+1} - p_t  (world-frame translation)
    Rotation delta:    q_t^{-1} * q_{t+1}  (rotation from t → t+1)
    """
    poses = np.asarray(poses, dtype=np.float32)
    p, q = poses[..., :3], _quat_normalize(poses[..., 3:7])
    dp = p[1:] - p[:-1]                                    # [T-1, 3]
    dq = _quat_mul(_quat_inv(q[:-1]), q[1:])               # [T-1, 4]
    return np.concatenate([dp, dq], axis=-1).astype(np.float32)


def delta_to_pose(pose0: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Inverse of pose_to_delta.  Integrates deltas [T-1, 7] from initial
    pose [7] to recover full trajectory [T, 7]."""
    pose0 = np.asarray(pose0, dtype=np.float32)
    deltas = np.asarray(deltas, dtype=np.float32)
    T = deltas.shape[0] + 1
    out = np.zeros((T, 7), dtype=np.float32)
    out[0] = pose0
    out[0, 3:7] = _quat_normalize(out[0, 3:7])
    for t in range(T - 1):
        out[t + 1, :3] = out[t, :3] + deltas[t, :3]
        out[t + 1, 3:7] = _quat_normalize(_quat_mul(out[t, 3:7], deltas[t, 3:7]))
    return out


# ──────────────────────────────────────────────────────────────────────
# Internal "flat" VQ-VAE training dataset (yields raw pose deltas)
# ──────────────────────────────────────────────────────────────────────

@dataclass
class FlatSample:
    text:    str
    deltas:  np.ndarray         # [T-1, 7]
    pose0:   np.ndarray         # [7]
    traj_id: str = ""


class FlatVQVAEDataset(Dataset):
    """Yields per-trajectory pose deltas + text + initial pose.

    Used by:
      • ``train.py --stage vqvae`` — to train FlatVQVAE on the deltas.
      • ``MotionGPTDataset`` (below) — wraps this so T5 fine-tune sees
        the same per-trajectory entries.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir:      Union[str, Path],
        split:         str = "train",
        T:             int = 30,
    ):
        self.T = int(T)
        self.entries: List[tuple] = []
        for traj_id, traj_dir, entry in iter_split_entries(
            manifest_path, data_dir, split,
        ):
            traj_npz = Path(traj_dir) / "trajectory.npz"
            if not traj_npz.exists():
                continue
            self.entries.append((traj_id, traj_dir, entry))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> FlatSample:
        traj_id, traj_dir, entry = self.entries[i]
        z = np.load(Path(traj_dir) / "trajectory.npz", allow_pickle=False)
        if "object_pose_world" not in z.files:
            # Fallback: zero-motion synthetic
            poses = np.zeros((self.T, 7), dtype=np.float32)
            poses[:, 6] = 1.0   # quaternion w=1
        else:
            poses = z["object_pose_world"].astype(np.float32)            # [T_orig, 7]
            T_orig = poses.shape[0]
            if T_orig < self.T:
                # Pad by repeating last pose
                pad = np.repeat(poses[-1:], self.T - T_orig, axis=0)
                poses = np.concatenate([poses, pad], axis=0)
            elif T_orig > self.T:
                # Uniform downsample
                idx = np.linspace(0, T_orig - 1, self.T).astype(int)
                poses = poses[idx]
        deltas = pose_to_delta(poses)                                    # [T-1, 7]

        # task_to_text expects (task_name, obj_category) — match the
        # canonical call site in dataload/dataset_a.py.
        if isinstance(entry, dict):
            text = task_to_text(
                entry.get("task_name", ""),
                entry.get("obj_category", ""),
            )
        else:
            text = str(entry)
        return FlatSample(
            text=text, deltas=deltas, pose0=poses[0],
            traj_id=traj_id,
        )


def collate_flat(samples: Sequence[FlatSample]) -> Dict:
    """Stack pose deltas from a batch of FlatSamples for VQ-VAE training."""
    deltas = np.stack([s.deltas for s in samples], axis=0)               # [B, T-1, 7]
    pose0  = np.stack([s.pose0  for s in samples], axis=0)               # [B, 7]
    return {
        "deltas":  torch.from_numpy(deltas).float(),
        "pose0":   torch.from_numpy(pose0).float(),
        "texts":   [s.text for s in samples],
        "traj_id": [s.traj_id for s in samples],
    }


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
    """Wraps :class:`FlatVQVAEDataset`; identical I/O semantics."""

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
