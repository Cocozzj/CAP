# PhysDreamer Baseline

Wrapper around [PhysDreamer](https://github.com/a1600012888/PhysDreamer)
(CVPR 2024) — physics-based interaction with 3D objects via video generation.

## Why PhysDreamer

In the new 5-baseline matrix, this fills the "4D physics generation" slot
(replacing 4D-GS which was reconstruction-only).

Paper angle: "PhysDreamer learns continuous physical fields from a video
diffusion prior; Ours predicts symbolic physical parameters that drive a
differentiable simulator.  Both produce 4D dynamics — we compare their
generalization across materials and tasks."

## Setup

```bash
# 1. Clone official repo
cd ~
git clone https://github.com/a1600012888/PhysDreamer.git
cd PhysDreamer

# 2. Install dependencies (Python 3.10 + CUDA 11.8+)
conda create -n physdreamer python=3.10 -y
conda activate physdreamer
pip install -r requirements.txt

# 3. Download pretrained video diffusion model (~10GB)
#    Follow PhysDreamer's README for the exact link
bash download_models.sh

# 4. Set environment variable
export PHYSDREAMER_REPO=~/PhysDreamer
```

## Comparison protocol

| Input | PhysDreamer | Ours |
|---|---|---|
| 3DGS scene | ✓ | ✓ |
| Text instruction | ✓ (sometimes; PD is mostly material+force) | ✓ |
| GT physics params | ❌ (learns from video prior) | ❌ (predicts from text) |

PhysDreamer doesn't require GT physics — it's a learned baseline like Ours.

| Metric | Expected |
|---|---|
| Visual quality | competitive (it has a strong video prior) |
| 4D consistency | competitive |
| Action conditioning | weaker (PD is more material-driven) |
| Cross-material | strong (it's designed for material variation) |
| Closure / Inverse | N/A (no algebraic structure) |

## Files (TODO — skeleton)

```
physdreamer/
├── __init__.py
├── README.md           this file
├── rho_to_config.py    our ρ tuple → PhysDreamer config dict
├── convert_data.py     iterate split, generate per-traj configs
└── run_eval.py         subprocess.run PhysDreamer → collect 4DGS outputs
```

## Implementation status

- [x] Skeleton + README
- [ ] `rho_to_config.py` — map our ρ to PhysDreamer's expected config
- [ ] `convert_data.py` — generate per-trajectory configs
- [ ] `run_eval.py` — call PhysDreamer's inference script
- [ ] Output parser (PhysDreamer writes its own format; convert to GS4DSequence)

## Estimated work

PhysDreamer integration: **1-2 weeks**.  Risks:
- Diffusion model weight download (~10GB) requires good bandwidth on lab server
- PhysDreamer's expected input format may need geometry preprocessing
- Inference is slow (diffusion sampling) — budget ~2-5 min per trajectory
