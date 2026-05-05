# MAGVIT v2 Baseline

Pixel-level video tokenizer baseline based on
[lucidrains' MAGVIT v2 implementation](https://github.com/lucidrains/magvit2-pytorch)
(unofficial, well-maintained).

## Why MAGVIT v2

It is the canonical **pixel-level video tokenizer**:
- VQ-VAE on (T, H, W, 3) videos → discrete tokens
- Transformer next-token prediction → autoregressive video generation

This baseline tests the hypothesis: **does our 3D-aware action representation
beat a flat pixel-level tokenizer on dynamic 3D scene generation?**

## Comparison protocol (MAGVIT v2 has NO 3D structure)

| Component               | MAGVIT v2          | Ours                 |
| ----------------------- | ------------------ | -------------------- |
| Tokenizer input         | RGB video frames   | 3DGS scene + actions |
| Codebook                | Flat, per-frame    | Hierarchical         |
| Multi-view consistency  | ❌ frame-by-frame   | ✅ inherent (3DGS)    |
| Physics-correct dynamics| ❌ pixel pattern    | ✅ via PBD            |
| Cross-object transfer   | ❌                 | ✅                    |

MAGVIT v2 takes a single-camera video (or per-camera independent generation)
and outputs a single-camera prediction. It has no notion of 3D state.

## What we measure

| Metric          | MAGVIT v2 | Ours |
| --------------- | --------- | ---- |
| Per-frame PSNR  | ✓         | ✓    |
| Per-frame LPIPS | ✓         | ✓    |
| FVD (video)     | ✓         | ✓    |
| Multi-view consistency error | high (independent per cam) | low (shared 3DGS) |
| 3D metrics (ADE/FDE) | N/A | ✓ |
| Closure / Inverse | N/A | ✓ |

The "multi-view consistency error" is the killer for MAGVIT v2 — generating
each cam independently leads to inconsistent appearances across views.

## Setup

```bash
# 1) Install lucidrains' MAGVIT v2
pip install magvit2-pytorch

# 2) (Optional) clone for examples
git clone https://github.com/lucidrains/magvit2-pytorch.git ~/magvit2-pytorch
```

The wrapper here doesn't shell out to a separate process — it uses the
`magvit2_pytorch` Python module directly.

## Training

Two stages:

```bash
# Stage 1: video tokenizer (3D causal CNN VQ-VAE)
python -m eval.baseline.magvit_v2.train --stage tokenizer --dataset a \
    --epochs 50 --output-dir runs/baselines/magvit_v2/dataset_a/tokenizer

# Stage 2: transformer for next-token prediction (text-conditioned)
python -m eval.baseline.magvit_v2.train --stage transformer --dataset a \
    --epochs 100 \
    --tokenizer-ckpt runs/baselines/magvit_v2/dataset_a/tokenizer/ckpt_final.pt \
    --output-dir runs/baselines/magvit_v2/dataset_a/transformer
```

Time: tokenizer ~6-10h, transformer ~2-4h (on 4× A100 with our 1650-sample
dataset).

## Inference

```bash
python -m eval.baseline.magvit_v2.infer \
    --tokenizer-ckpt   runs/baselines/magvit_v2/dataset_a/tokenizer/ckpt_final.pt \
    --transformer-ckpt runs/baselines/magvit_v2/dataset_a/transformer/ckpt_final.pt \
    --manifest dataset/dataset_a/manifest.json \
    --data-dir dataset/dataset_a/data \
    --splits   test_iid \
    --output-root runs/baselines
```

This produces `pred_render.mp4` per trajectory (no `pred_4dgs.npz` —
MAGVIT v2 is pixel-only).

## Caveats

- Per-camera training: trains 3 independent MAGVIT v2's (one per cam0/1/2)
  for Dataset-A.  This is the standard MAGVIT setup.  Multi-view consistency
  is not enforced.
- For Dataset-B (single cam): trains one MAGVIT v2 on cam0.mp4.
- Resolution: we down-sample to 128×128 for compute (MAGVIT v2 at 256×256 is
  ~4× slower).  PSNR comparisons are at this resolution; mention in paper.
- Token sequence length grows with T × H/8 × W/8 — at T=30, 128×128
  resolution, that's ~7680 tokens.  Transformer needs flash attention or
  sliding-window attention to handle this.
