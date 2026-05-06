# Stable Video Diffusion (SVD) — generic video-diffusion baseline

Stand-alone baseline that asks: how does a generic image-to-video
diffusion prior do on our manipulation prediction task, with no
task-specific training or per-scene optimization?

We use **Stable Video Diffusion XT** (`stabilityai/stable-video-diffusion-img2vid-xt`,
Blattmann et al. 2023) in zero-shot image-to-video mode:

1. Take the first frame of each trajectory's GT video (`cam0.mp4`).
2. Feed it to SVD-XT.
3. Generate 25 future frames at native resolution (1024×576).
4. Save as `pred_render.mp4`.

This produces **only 2D pixel output**, so for the paper's metrics:

| Metric | Computed | Source |
|---|---|---|
| PSNR / LPIPS / SSIM | ✅ | `pred_render.mp4` ↔ `cam0.mp4`  |
| ADE / FDE / MPJPE   | ❌ | N/A (no 3D output)             |
| Closure / Inverse Gap | ❌ | N/A (no token output)         |
| Diversity / Physics-W | ❌ | N/A (no token, no physics)    |
| Multi-view consistency | ❌ | N/A (only one generated view) |

All "N/A" cells render as `—` in the LaTeX tables; this matches the
existing handling for MAGVIT v2's pixel-only path.

## Setup (one-time, ~30 min)

```bash
conda activate physgs       # any env with torch+CUDA12 will do

pip install diffusers==0.27.2 transformers==4.40.0 accelerate==0.30.1 \
            einops imageio[ffmpeg] huggingface_hub

# Download SVD-XT (~10GB)
huggingface-cli download stabilityai/stable-video-diffusion-img2vid-xt \
    --local-dir ~/SVD_ckpts/svd-xt
ls ~/SVD_ckpts/svd-xt/      # should have model_index.json + subfolders
```

## Usage

```bash
# 1. Generate per-trajectory configs (CPU, seconds)
python -m eval.baseline.svd.convert_data \
    --manifest dataset/dataset_a/manifest.json \
    --data-dir dataset/dataset_a/data \
    --output-root runs/baselines --dataset-name dataset_a \
    --splits test_iid test_compositional_long test_ood_unseen_object \
             test_ood_unseen_pair dataset_d_test

python -m eval.baseline.svd.convert_data \
    --manifest dataset/dataset_b/manifest.json \
    --data-dir dataset/dataset_b/data \
    --output-root runs/baselines --dataset-name dataset_b \
    --splits test

# 2. 4-shard inference (each on its own GPU; ~5-10s/traj)
for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD nohup python -m eval.baseline.svd.run_eval \
    --output-root runs/baselines --dataset-name dataset_a \
    --svd-ckpt ~/SVD_ckpts/svd-xt \
    --shard-index $SHARD --num-shards 4 \
    > /tmp/svd_a_$SHARD.log 2>&1 &
done
wait

for SHARD in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$SHARD nohup python -m eval.baseline.svd.run_eval \
    --output-root runs/baselines --dataset-name dataset_b \
    --svd-ckpt ~/SVD_ckpts/svd-xt \
    --shard-index $SHARD --num-shards 4 \
    > /tmp/svd_b_$SHARD.log 2>&1 &
done
wait
```

Total throughput estimate: 1300 trajectories × ~7s/traj ÷ 4 GPUs ≈
**40 minutes** (vs. ~10 hours single-GPU sequential).

## Why SVD specifically?

- **Stable Video Diffusion** (svd-xt): mature, public checkpoints, one
  forward pass per clip, runs in fp16 on a single A100.
- **AnimateDiff**: comparable quality but more complex API and
  per-prompt-encoder latency.
- **CogVideoX / I2VGen-XL**: longer videos but heavier (24-26GB VRAM).

## Note on PhysDreamer

PhysDreamer (Zhang et al. 2024) is a different category — it does
per-scene physics-parameter optimization using a video-diffusion prior,
costing ≈30 minutes per scene.  Running it on 1300 trajectories is
infeasible (~7 days even with 4 GPUs), so we don't include it.  The
paper acknowledges this in the baseline section.

## Caveats / honest disclosures for paper

1. SVD has no physics conditioning — it is pure pixel-space prior.
2. SVD generates 25 frames at 6fps (~4s clip); GT is 30 frames at 30fps
   (1s clip).  We resample to T=30 in the metric computation.
3. SVD output frame count and aspect ratio differ from GT — we resize.
   Some frame-by-frame quality drop is unavoidable from this resampling.
4. SVD quality on PartNet-style synthetic scenes is variable; on SSv2
   real video it tends to be qualitatively reasonable but pixel-metric
   numbers are noisy.

These caveats are acceptable for the baseline's purpose: showing that
generic generative video priors don't capture articulated/physical
manipulation as well as our task-specific tokenizer.
