"""External baselines for the paper's main results table.

4-baseline matrix (paper §4 main table):

  tamp_pddl       Symbolic decompose + motion primitives (PDDLStream
                  itself wasn't installable on the eval cluster — the
                  fallback is fully fair, see tamp_pddl/README.md)
  physgaussian    Differentiable MPM physics simulator (no training)
  svd             Stable Video Diffusion zero-shot image-to-video —
                  generic generative video prior baseline.  Pixel-only
                  output, so 3D metrics report N/A.  See svd/README.md.
  motiongpt       T5 + motion VQ tokens (fine-tuned on our data) —
                  motion-level tokenization baseline.

Two baselines were *intentionally excluded*:
  • PhysDreamer (Zhang et al. 2024) — per-scene optimization at
    ~30 min/scene is infeasible across our 1300+ evaluation trajectories.
  • MAGVIT-v2 (Yu et al. 2023) — the public implementation
    (``lucidrains/magvit2-pytorch``) has an unstable transformer-side
    API.  The pixel-level video baseline niche is already covered by
    SVD, so the marginal value of debugging MAGVIT-v2 was low.

Plus:

  ours            Our method, integrated via eval.baseline.ours.runner

Aggregation:

  common.py            Shared I/O format (GS4DSequence + TrajMetrics)
  metrics.py           Per-trajectory geometric metrics
  aggregate.py         Cross-trajectory aggregation → main_table.json
  render_metrics.py    Visual metrics (PSNR / LPIPS / SSIM)
  diversity_eval.py    PDF #9 (Levenshtein, multi-sample)
  physics_wasserstein  PDF #11 (trajectory physics deviation)
  length_curve_eval    PDF #8 (length-vs-success curve + plot)
  format_latex.py      Paper Table 1 + Table 2 LaTeX output

Deprecated (kept for backward-compat reference):

  physdreamer/    superseded by svd/ (renamed module).  The directory
                  may still exist in old branches; the new module is
                  eval.baseline.svd
  tamp_rule       superseded by tamp_pddl
  _4dgs           reconstruction-only; not in main table
  flat_vqvae      moved to ablation section (not external baseline)
  t2m_gpt         renamed to flat_vqvae long ago
"""
