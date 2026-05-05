# PhysGaussian Baseline

Wrapper around [PhysGaussian](https://github.com/XPandora/PhysGaussian)
(CVPR 2024) — differentiable MPM physics on 3DGS scenes.

## Why PhysGaussian

It is the closest competitor to our physics module:

- Operates on 3DGS scenes (same data format as ours)
- Uses MPM-style continuum physics (similar to our PBD soft-body backend)
- Has no learnable parameters → pure simulator (no training)

## Comparison protocol

| Input          | PhysGaussian              | Ours                          |
| -------------- | ------------------------- | ----------------------------- |
| Scene          | init_gs.ply               | init_gs.ply                   |
| Physics params | **GT material params**    | predicted from text           |
| Conditioning   | None (open-loop sim)      | text instruction              |

PhysGaussian gets GT physics params (its UNFAIR advantage). We measure:

- **Trajectory error** (ADE/FDE/MPJPE) on the deformed Gaussians
- **Visual quality** (PSNR/LPIPS) on rendered video
- **Physics consistency** (energy / contact / volume drift)

PhysGaussian should win on physics-consistency and lose on text-conditioning
(it has no text input → cannot operate from text alone).

## Setup (run on lab server)

```bash
# 1) Clone the official repo
cd ~
git clone https://github.com/XPandora/PhysGaussian.git
cd PhysGaussian

# 2) Create conda env (isolated from main pipeline)
conda create -n physgs python=3.10 -y
conda activate physgs

# 3) Install dependencies
pip install torch==2.0.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install taichi

# 4) Test with their demo
bash download_model.sh
python gs_simulation.py \
    --model_path model/wolf \
    --output_path debug_output/

# 5) Set environment variable so our wrapper finds it
export PHYSGAUSSIAN_REPO=~/PhysGaussian
```

## Running on our data

```bash
# Build PhysGaussian configs from our ρ tuples (one per trajectory)
python -m eval.baseline.physgaussian.convert_data \
    --manifest dataset/dataset_a/manifest.json \
    --data-dir dataset/dataset_a/data \
    --splits test_iid test_ood_unseen_pair \
    --output-root runs/baselines/physgaussian/dataset_a \
    --T 30

# Run PhysGaussian on every trajectory
python -m eval.baseline.physgaussian.run_eval \
    --output-root runs/baselines/physgaussian/dataset_a \
    --physgs-repo $PHYSGAUSSIAN_REPO
```

## ρ → PhysGaussian material mapping

PhysGaussian supports these material presets:

| PhysGaussian preset | Closest match for our ρ tuple |
| ------------------- | ----------------------------- |
| `jelly`             | low E (< 1e6), high ν (> 0.3), low ρ_m (< 500) |
| `metal`             | high E (> 1e10), low-mid ν (~0.3), high ρ_m   |
| `sand`              | (not used by PartNet articulation, not mapped) |
| `foam`              | low E (~1e5), low ν (< 0.3), very low ρ_m     |
| `plasticine`        | mid E (~1e7), low ν (~0.3), mid ρ_m           |

PartNet-Mobility objects are mostly **rigid** (drawers, doors, handles).
PhysGaussian doesn't have a pure-rigid preset; we use `metal` (highest E)
as the closest proxy. See `rho_to_config.py`.

## Caveats

- PhysGaussian uses MPM (Material Point Method); our PBD backend is
  different but related. Comparing them on the same trajectory is fair.
- PhysGaussian's solver may diverge on certain articulated objects (e.g.
  thin doors). The wrapper catches solver failures and marks those
  trajectories as `solver_diverged` in metrics.json.
- For Dataset-B (real video), PhysGaussian still runs but ρ comes from
  PartNet-style heuristics since SS-v2 has no GT physics params.
