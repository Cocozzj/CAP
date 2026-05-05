"""DataLoader package for CAP.

Two datasets, single shared utility module:

  DatasetA    PartNet-Mobility synthetic, 3-camera mp4, GT physics (unused)
  DatasetB    Something Something v2 real video, 1-camera mp4, MiDaS depth

Both produce per-sample dicts compatible with the same ``collate_fn`` —
optional fields (``intrinsics`` / ``extrinsics`` / ``task_id`` / ``depth``)
are included only when present, so ``model.forward`` and the trainer
work with either dataset without branching.

Quick start::

    from dataload import DatasetA, DatasetB, collate_batch
    ds  = DatasetA("data/dataset_a/manifest.json", "data/dataset_a/data", split="train")
    ldr = DataLoader(ds, batch_size=8, collate_fn=collate_batch)
"""

from .common import (
    _load_video_frames,
    collate_batch,
    collate_fn,
    load_cameras,
    load_init_gs_ply,
)
from .dataset_a import DatasetA
from .dataset_b import DatasetB
from .text import dataset_b_text, task_to_text

__all__ = [
    # Datasets
    "DatasetA", "DatasetB",
    # Collate
    "collate_fn", "collate_batch",
    # Loaders (in case eval scripts want to call directly)
    "load_init_gs_ply", "load_cameras",
    # Text helpers
    "task_to_text", "dataset_b_text",
]
