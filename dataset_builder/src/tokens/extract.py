from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RHO_DIM = 8


@dataclass(frozen=True)
class ActionTokens:
    ell: np.ndarray
    h: np.ndarray
    xi: np.ndarray
    rho: np.ndarray
    token_id: np.ndarray
    c_task_id: np.ndarray
    timestamps: np.ndarray


def empty_action_tokens(seq_len: int = 12, c_task_id: int = -1) -> ActionTokens:
    return ActionTokens(
        ell=np.zeros((seq_len, 3), dtype=np.float32),
        h=np.zeros((seq_len, 4), dtype=np.float32),
        xi=np.zeros((seq_len, 6), dtype=np.float32),
        rho=np.zeros((seq_len, RHO_DIM), dtype=np.float32),
        token_id=np.full((seq_len,), -1, dtype=np.int32),
        c_task_id=np.asarray(c_task_id, dtype=np.int32),
        timestamps=np.linspace(0.0, 1.0, seq_len, endpoint=False, dtype=np.float32),
    )


def save_action_tokens(path: str, tokens: ActionTokens) -> None:
    np.savez_compressed(
        path,
        ell=tokens.ell,
        h=tokens.h,
        xi=tokens.xi,
        rho=tokens.rho,
        token_id=tokens.token_id,
        c_task_id=tokens.c_task_id,
        timestamps=tokens.timestamps,
    )
