# Side-by-Side Qualitative Figure — Workflow

Generates the NeurIPS main-paper qualitative comparison figure
(2 tasks × N methods × T timesteps grid).

## Two-stage pipeline

```
[1] render_rollout_frames.py      [2] figure_qualitative_comparison.py
       roll out each method     →        compose grid PDF
       render to PNG frames              add Ours highlight, failure boxes
       (one per timestep)                add ✓/✗ indicators
```

## Output structure

```
runs/figures/qualitative/
    frames/                                  # ← stage 1 output
        open_drawer_unseen/
            Ours/         t000.png  t002.png ...
            PhysGaussian/ t000.png  t002.png ...
            SVD/          t000.png  t002.png ...
            MotionGPT/    t000.png  t002.png ...
        long_horizon_5step/
            ...
    failures.json                            # ← describes which frames failed
    fig_qualitative.pdf                      # ← stage 2 output (paper figure)
```

---

## Stage 1: Render rollout frames

For your trained models:

```bash
python -m eval.viz.render_rollout_frames \
    --tasks open_drawer_unseen:"open the drawer" \
            long_horizon_5step:"open then push then rotate" \
    --methods Ours:runs/main_exp/seed_0/ckpt/main_exp_final.pt \
              PhysGaussian:runs/baseline_physgaussian/ckpt/final.pt \
              SVD:runs/baseline_svd/ckpt/final.pt \
              MotionGPT:runs/baseline_motiongpt/ckpt/final.pt \
    --timesteps 0 2 4 6 8 10 \
    --image-size 256 256 \
    --output-dir runs/figures/qualitative/frames
```

Notes:
- Use the **same camera** for all (task, method) pairs to ensure fair
  side-by-side comparison.  Pass `--camera path/to/camera.yaml` if you
  have a saved viewpoint config.
- Without `--camera` the script uses a default front view.
- If `gsplat` / `nerfacc` is not installed, falls back to a 2D scatter
  projection so the layout still works.

For external baselines that don't share Act4D's checkpoint format
(e.g., PhysGaussian, SVD), you'll need a per-baseline render script
that produces the same `<task>/<method>/t???.png` structure.

---

## Stage 2: Define failures (optional but recommended)

Create `failures.json`:

```json
{
  "open_drawer_unseen": {
    "Ours":         { "result": "success", "failed_steps": [] },
    "PhysGaussian": { "result": "fail",
                      "failed_steps": [4, 6, 8, 10],
                      "annotation": "Object distorted at t=4" },
    "SVD":          { "result": "partial",
                      "failed_steps": [8, 10],
                      "annotation": "Drift at t=8" },
    "MotionGPT":    { "result": "fail",
                      "failed_steps": [4, 6, 8, 10],
                      "annotation": "Joint-space jump" }
  },
  "long_horizon_5step": {
    "Ours":         { "result": "success", "failed_steps": [] },
    "PhysGaussian": { "result": "fail",
                      "failed_steps": [6, 8, 10],
                      "annotation": "Cumulative error" },
    "SVD":          { "result": "fail",
                      "failed_steps": [4, 6, 8, 10],
                      "annotation": "AR drift" },
    "MotionGPT":    { "result": "partial",
                      "failed_steps": [8, 10] }
  }
}
```

Fields:
- `result`: `"success"` / `"fail"` / `"partial"` / `"unknown"`
- `failed_steps`: list of timesteps to draw a red border around
- `annotation`: optional short text shown beside the indicator

---

## Stage 3: Compose the figure

```bash
python -m eval.viz.figure_qualitative_comparison \
    --frames-dir runs/figures/qualitative/frames \
    --tasks "open_drawer_unseen:Cross-object transfer ('open' learned on cabinet → unseen drawer)" \
            "long_horizon_5step:Long-horizon composition (5-step plan)" \
    --methods Ours PhysGaussian SVD MotionGPT \
    --timesteps 0 2 4 6 8 10 \
    --failures runs/figures/qualitative/failures.json \
    --output runs/figures/qualitative/fig_qualitative.pdf
```

---

## Quick demo (preview layout without real renders)

To validate the layout while waiting for actual model rollouts:

```bash
python -m eval.viz.figure_qualitative_comparison \
    --frames-dir /tmp/demo_frames \
    --tasks "open_drawer_unseen:Cross-object transfer (demo)" \
            "long_horizon_5step:Long-horizon composition (demo)" \
    --methods Ours PhysGaussian SVD MotionGPT \
    --timesteps 0 2 4 6 8 10 \
    --output /tmp/demo_qualitative.pdf \
    --demo
```

This generates synthetic placeholder frames + a default failure spec,
so you can see the final layout immediately.

---

## Embedding in LaTeX

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\linewidth]{Figure/fig_qualitative.pdf}
  \caption{
    \textbf{Side-by-side qualitative comparison.}
    Top: cross-object transfer of an ``open'' action learned on cabinet
    doors and applied to an unseen drawer. Bottom: a 5-step long-horizon
    composition. ATOM (highlighted row) executes both tasks correctly,
    while PhysGaussian fails to generalize to the unseen geometry, SVD
    becomes geometrically inconsistent at $t\!\geq\!8$, and MotionGPT
    produces jumpy motion as its joint-space tokens are not grounded in
    object geometry. Red borders mark failed frames; \textcolor{green!60!black}{\ding{51}}
    /\textcolor{red}{\ding{55}} on the right indicate task success at $t=10$.
  }
  \label{fig:qualitative}
\end{figure*}
```

Required LaTeX packages:
```latex
\usepackage{pifont}      % for ✓ / ✗
\usepackage{xcolor}      % for green / red
```

---

## Customization knobs

In `figure_qualitative_comparison.py`:

| Constant       | Default               | Effect                                    |
|----------------|-----------------------|-------------------------------------------|
| `OURS_TINT`    | `(0.85, 1.0, 0.85)`   | Background tint on the "Ours" row         |
| `FAIL_COLOR`   | `#D62728` (red)       | Border color for failed frames            |
| `SUCCESS_COLOR`| `#2CA02C` (green)     | ✓ marker color                            |
| `FAIL_INDICATOR`| `#D62728` (red)      | ✗ marker color                            |
| `PARTIAL_COLOR`| `#FF7F0E` (orange)    | ⚠ marker color                            |

CLI flags:
- `--cell-inches 0.9`  — per-cell width/height in inches
- `--dpi 200`          — output resolution

---

## Tips for paper-ready figures

1. **Keep all images at the same aspect ratio and pixel size.**
   `render_rollout_frames.py` enforces this via `--image-size`.

2. **Use the same camera viewpoint** across methods.  Side-by-side comparison
   only works if the camera doesn't move between methods.

3. **Highlight differences explicitly** via `failures.json`.  Don't hope the
   reviewer will spot the broken frame on their own — circle it red.

4. **Pick 2 strong tasks** that demonstrate Act4D's unique strengths:
   - Task 1: cross-object transfer (most baselines crash)
   - Task 2: long-horizon composition (autoregressive baselines drift)

5. **Caption explains the layout in the first sentence** — reviewer should
   know what to look at within 5 seconds.
