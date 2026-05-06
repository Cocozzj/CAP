"""Evaluation suite for the CAP main experiment.

Each module corresponds to one inference protocol from the PDFs:

  text_conditioned.py     — Mode A: text → action  (PDF §3.5)
  imitation.py            — Mode B: demo video → action
  composite.py            — Mode C: composite task (PDF §4.1 Prop 4)
  physics_counterfactual.py — Cross-material/mass/friction ablation
                              (PDF f07d2c0a §1.1 + fdfa011c 行 1101)

Each script can be run standalone:
    python -m eval.text_conditioned --ckpt runs/main_exp/ckpt/main_exp_final.pt

NOTE: We deliberately do NOT eagerly import ``metrics.py`` here.  Doing so
would pull in the full ``model.*`` chain (transformers, timm, lpips, ...)
even when running lightweight wrappers like ``eval.baseline.physgaussian``
that only need numpy / json / plyfile.  Each script imports what it needs.
"""
__all__: list = []
