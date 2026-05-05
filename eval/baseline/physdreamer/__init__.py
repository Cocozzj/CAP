"""PhysDreamer wrapper — 4D physics generation baseline.

Wraps the official `PhysDreamer <https://github.com/a1600012888/PhysDreamer>`_
(CVPR 2024) inference code.  PhysDreamer takes a static 3DGS scene + a video
diffusion prior and produces a 4D physical interaction sequence.

Paper positioning: in the 5-baseline matrix this represents
"4D physics generation" — the closest direct competitor to Ours' physics
prediction module.

Submodules:
  rho_to_config   map our ρ tuple → PhysDreamer config
  convert_data    write per-traj PhysDreamer config files
  run_eval        invoke PhysDreamer + collect outputs → pred_4dgs.npz
"""
