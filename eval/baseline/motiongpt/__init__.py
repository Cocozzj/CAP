"""MotionGPT wrapper — discrete action token + LLM baseline.

Wraps the official `MotionGPT <https://github.com/OpenMotionLab/MotionGPT>`_
(NeurIPS 2023) which treats human motion as a foreign language: pre-trained
LLM (T5/GPT) predicts motion tokens from text.

Paper positioning: in the 5-baseline matrix, MotionGPT represents
"discrete motion tokens + LLM" — a strong learned baseline that contrasts
with our hierarchical + group-algebraic structure.

Submodules:
  data    pose-delta dataset matching MotionGPT's expected I/O
  train   fine-tune MotionGPT on our action-token vocabulary
  infer   text → tokens → 4DGS via Ours' renderer
"""
