"""Visualization utilities for the CAP paper figures.

Modules
-------
qualitative_videos
    V.1 figure — 5 tasks × 6 methods grid of trajectory snapshots
    (live rollout from checkpoints, scatter or 3DGS render).
render_rollout_frames
    Stage 1 of the side-by-side qualitative pipeline: roll out each
    (task, method) pair and dump fixed-camera PNG frames.
figure_qualitative_comparison
    Stage 2 of the side-by-side qualitative pipeline: compose the
    pre-rendered frames into a NeurIPS-ready 2-task × N-method ×
    T-timestep grid with Ours highlight, failure boxes, success
    indicators.
tsne
    Fig 4 — t-SNE of the atomic / task codebooks.

See QUALITATIVE_README.md for the full workflow.
"""
