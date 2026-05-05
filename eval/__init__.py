"""Evaluation suite for the CAP main experiment.

Each module corresponds to one inference protocol from the PDFs:

  text_conditioned.py     — Mode A: text → action  (PDF §3.5)
  imitation.py            — Mode B: demo video → action
  composite.py            — Mode C: composite task (PDF §4.1 Prop 4)
  physics_counterfactual.py — Cross-material/mass/friction ablation
                              (PDF f07d2c0a §1.1 + fdfa011c 行 1101)

Each script can be run standalone:
    python -m eval.text_conditioned --ckpt runs/main_exp/ckpt/main_exp_final.pt
"""

from .metrics import psnr, lpips_score, scene_distance_metric

__all__ = ["psnr", "lpips_score", "scene_distance_metric"]
