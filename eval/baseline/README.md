# Baseline Methods for Main Results Table

This directory contains the **5-baseline matrix** compared against Ours in
the paper's main quantitative table (PDF §A1 / §A2).

| Baseline       | Class                            | Contrasts (vs Ours) |
| -------------- | -------------------------------- | -------------------- |
| `tamp_pddl/`   | Classical TAMP (PDDLStream)      | learning vs hand-written rules |
| `physgaussian/`| MPM physics simulator + 3DGS     | action semantics (it has none) |
| `physdreamer/` | 4D physics generation (diffusion)| symbolic control (it's continuous field) |
| `magvit_v2/`   | Pixel-level video tokenizer      | 3D structure understanding |
| `motiongpt/`   | Discrete motion tokens + LLM     | hierarchy + group-algebraic structure |

Plus `ours/` — our method, integrated via the same I/O format so the
aggregator pulls Ours into the same tables.

## Common output format

Every baseline writes per-trajectory results to:

```
runs/baselines/<baseline_name>/<dataset>/<split>/<traj_id>/
    pred_4dgs.npz      # predicted 4DGS sequence (for non-pixel baselines)
    pred_render.mp4    # rendered video (V cameras concatenated; pixel baselines)
    metrics.json       # per-trajectory metrics
```

`pred_4dgs.npz` schema:

```python
{
    "mu":      np.float32[T, N, 3],          # gaussian centers
    "cov":     np.float32[T, N, 3, 3],       # gaussian covariances
    "sh":      np.float32[T, N, 48],         # spherical harmonics (degree 3 RGB)
    "opacity": np.float32[T, N, 1],          # alpha
    "scale":   np.float32[T, N, 3],          # log-scale per axis
    "T":       int,                           # number of timesteps
}
```

Pixel-only baselines (MAGVIT-v2) skip `pred_4dgs.npz` and write only
`pred_render.mp4`.

## Aggregation pipeline

```bash
# 1. Run each baseline's inference (TAMP / Phys / PhysDreamer / MAGVIT / MotionGPT / Ours)
python -m eval.baseline.tamp_pddl.run_tamp ...
python -m eval.baseline.physgaussian.run_eval ...
python -m eval.baseline.physdreamer.run_eval ...
python -m eval.baseline.magvit_v2.infer ...
python -m eval.baseline.motiongpt.infer ...
python -m eval.baseline.ours.runner ...

# 2. Geometric metrics (ADE / FDE / Success / Energy)
python -m eval.baseline.aggregate \
    --baselines tamp_pddl physgaussian physdreamer magvit_v2 motiongpt ours \
    --datasets dataset_a dataset_b \
    --output runs/main_table.json

# 3. Visual metrics (PSNR / LPIPS / SSIM, requires gsplat)
python -m eval.baseline.render_metrics

# 4. Diversity (PDF metric #9, multi-sample Levenshtein)
python -m eval.baseline.diversity_eval --N 10

# 5. Physics Wasserstein (PDF metric #11)
python -m eval.baseline.physics_wasserstein --method wasserstein

# 6. Length-vs-success curve (PDF metric #8)
python -m eval.baseline.length_curve_eval

# 7. Re-aggregate to merge all metrics
python -m eval.baseline.aggregate --baselines ... --output runs/main_table.json

# 8. LaTeX paper tables
python -m eval.baseline.format_latex \
    --json runs/main_table.json --dataset dataset_a --two-tables \
    --output1 runs/table1_reliability.tex \
    --output2 runs/table2_diversity_physics.tex
```

## Per-baseline READMEs

See each `<baseline>/README.md` for setup + comparison angle:

- `tamp_pddl/README.md`     — PDDLStream setup, domain.pddl, motion primitives
- `physgaussian/README.md`  — clone official PhysGaussian repo, ρ → config mapping
- `physdreamer/README.md`   — clone PhysDreamer repo, video diffusion model download
- `magvit_v2/README.md`     — lucidrains' magvit2-pytorch, video VQ tokenizer
- `motiongpt/README.md`     — clone OpenMotionLab/MotionGPT, T5 fine-tuning
- `ours/README.md`          — our method's adapter (see `ours/runner.py`)

## Where each baseline appears in the paper

| Experiment              | TAMP-PDDL | PhysG | PhysDream | MAGVIT | MotGPT |
| ----------------------- |:---------:|:-----:|:---------:|:------:|:------:|
| §4 Table 1 (main)       |    ✓      |   ✓   |    ✓      |   ✓    |   ✓    |
| §4 Table 2 (diversity)  |    —      |   —   |    ✓      |   ✓    |   ✓    |
| §4 long-horizon         |    ✓      |   ✓   |    ✓      |   ✓    |   ✓    |
| §4 real-data (B)        |   weak    | weak  |    ✓      |   ✓    |   ✓    |
| §4 cross-material       |    —      |   ✓   |    ✓      |   —    |   —    |

## Deprecated baselines

The following exist as deprecation stubs only — kept so external scripts
don't break, but no new development.  Safe to delete:

- `tamp_rule/`  — superseded by `tamp_pddl/`
- `_4dgs/`      — reconstruction not generation; removed from main table
- `t2m_gpt/`    — renamed to `flat_vqvae/` long ago

`flat_vqvae/` itself is now an ABLATION baseline (paper §6, Table 4),
not an external baseline.  See `flat_vqvae/README.md`.
