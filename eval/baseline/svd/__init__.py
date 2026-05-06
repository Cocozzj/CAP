"""Stable Video Diffusion (SVD) — generic video-diffusion baseline.

Tests how a generic image-to-video diffusion prior (Blattmann et al.
2023, ``stabilityai/stable-video-diffusion-img2vid-xt``) performs on
our manipulation/articulation prediction task in zero-shot mode:

  1. Take the first frame of each trajectory's GT video (cam0.mp4)
  2. Feed it to SVD to generate a 25-frame extrapolation
  3. Save as ``pred_render.mp4`` next to the config

This is a *pixel-space generative prior* baseline (no physics, no 3D,
no scene model), included to answer the natural reviewer question
"how does a generic video diffusion model do?".  Output is 2D-only:
  - 2D metrics (PSNR / LPIPS / SSIM) — computed via render_metrics.py's
    ``is_pixel_only`` path (same machinery as MAGVIT v2)
  - 3D metrics (ADE / FDE / MPJPE / Closure Gap / Inverse Gap) — N/A;
    aggregate / format_latex render a dash in those cells

Files:
  convert_data.py   — enumerate trajectories, write svd_config.json
  run_eval.py       — load first frame, run SVD, write pred_render.mp4
  README.md         — install / setup notes
"""
