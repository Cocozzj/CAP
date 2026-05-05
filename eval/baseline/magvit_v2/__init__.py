"""MAGVIT v2 baseline (pixel-level video VQ + transformer).

Wraps lucidrains' magvit2-pytorch implementation.  Two-stage pipeline:
  Stage 1: 3D causal CNN VQ-VAE on RGB videos
  Stage 2: Transformer next-token prediction

Output is pixel video (pred_render.mp4) — no 3DGS / 3D structure.
"""
