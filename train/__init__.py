"""CAP training package.

Layout:
    stages.py   — StageSpec dataclass + curriculum presets
                    DEFAULT_STAGES   (Dataset-A 4-stage 150ep)
                    SMOKE_STAGES     (1 epoch / stage variant of A)
                    DATASET_B_STAGES (Dataset-B single 30ep fine-tune)
                    SMOKE_STAGES_B   (1 epoch / stage variant of B)
    trainer.py  — main entry point (run with ``python -m train.trainer ...``)

Re-exports the curriculum data so callers can do
``from train import StageSpec, DATASET_B_STAGES``.
"""
from .stages import (
    StageSpec, DEFAULT_STAGES, SMOKE_STAGES, DATASET_B_STAGES, SMOKE_STAGES_B,
)

__all__ = [
    "StageSpec", "DEFAULT_STAGES", "SMOKE_STAGES",
    "DATASET_B_STAGES", "SMOKE_STAGES_B",
]
