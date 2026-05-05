# MotionGPT Baseline

Wrapper around [MotionGPT](https://github.com/OpenMotionLab/MotionGPT)
(NeurIPS 2023) — text → discrete motion tokens via pretrained LLM.

## Why MotionGPT

Direct competitor in the "language model + discrete action tokens" space.
Compared to our self-implemented `flat_vqvae` (which uses a small custom
GPT), MotionGPT uses a **pretrained T5** with stronger language priors.

Paper angle: "MotionGPT achieves strong text-to-motion alignment via LLM
pretraining but lacks 3D scene awareness, hierarchical structure, and
algebraic guarantees — Ours wins on all three axes."

## Setup

```bash
# 1. Clone official repo
cd ~
git clone https://github.com/OpenMotionLab/MotionGPT.git
cd MotionGPT
conda env create -f environment.yml
conda activate mgpt

# 2. Download pretrained T5 + motion VQ-VAE checkpoints
bash download_motiongpt.sh    # follow their README

# 3. Set environment variable
export MOTIONGPT_REPO=~/MotionGPT
```

## Comparison protocol

| Component | MotionGPT | Ours |
|---|---|---|
| Text encoder | T5 (pretrained, ~220M params) | Sentence-Transformer |
| Motion tokens | flat VQ codebook (K=512) | hierarchical (atomic + task) |
| LLM | T5 + finetune | CVAE + AR |
| 3D scene | ❌ ignored | ✓ conditions on init_gs |
| Algebraic | ❌ none | ✓ closure / inverse / equiv |

## Files (TODO — skeleton)

```
motiongpt/
├── __init__.py
├── README.md           this file
├── data.py             pose-delta dataset matching MotionGPT's HumanML3D format
├── train.py            fine-tune MotionGPT on our token vocabulary
└── infer.py            text → MotionGPT → tokens → 4DGS via Ours' renderer
```

## Implementation plan

1. **Data adaptation**: convert our action-token sequences to MotionGPT's
   "motion-as-language" format (token IDs + special tokens for sentence
   boundaries).
2. **Vocab extension**: add our K=64 atomic + J=128 task tokens to MotionGPT's
   T5 tokenizer as new tokens.
3. **Fine-tune**: continue pre-training on our (text → token sequence) pairs.
4. **Inference**: T5 generates token sequence → decode through our VQ-VAE
   embedding → integrate via our Executor → 4DGS.

## Estimated work

MotionGPT integration: **1 week**.  Risks:
- T5 + new vocabulary requires careful tokenizer extension
- MotionGPT's training script assumes HumanML3D format; needs adaptation
- Fine-tuning on small data (~1650 samples) may need careful learning rate
