"""CAP training package.

Layout:
    stages.py   — StageSpec dataclass + DEFAULT_STAGES + SMOKE_STAGES presets
    trainer.py  — main entry point (run with ``python -m train.trainer ...``)

Re-exports the curriculum data so callers can do ``from train import StageSpec, DEFAULT_STAGES``.
"""
from .stages import StageSpec, DEFAULT_STAGES, SMOKE_STAGES

__all__ = ["StageSpec", "DEFAULT_STAGES", "SMOKE_STAGES"]
