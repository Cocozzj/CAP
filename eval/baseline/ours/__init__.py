"""Ours' inference adapter for the baseline pipeline.

Loads our trained CAPModel ckpt, runs ``infer_text`` per test trajectory,
writes pred_4dgs.npz in the standard baseline output format so the
aggregator and format_latex.py can pick it up.

Submodules:
  runner    main entry — iterate split → write per-trajectory outputs
"""
