"""
Stage schedule for the 4-stage training curriculum.

Stages
──────────────────────────────────────────
  RIGID    35 epochs   lr=3e-4    Encoder + Executor (rigid path) trainable;
                                  Planner frozen but forward computed (NCE warmup);
                                  physics OFF.
  PLANNER  35 epochs   lr=2e-4    Planner trainable (CVAE + AR);
                                  Encoder + Executor frozen;
                                  physics OFF; CVAE β + L_comm anneal start here.
  PHYSICS  25 epochs   lr=1e-4    Executor.deform only; rest frozen;
                                  physics ON, lipschitz + physics_loss on.
  FULL     55 epochs   lr=3e-5    Everything trainable, full loss suite,
                                  scheduled-sampling 0→0.5 over first 15 ep.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List

from model import LossSpec

# ══════════════════════════════════════════════════════════════════════
# StageSpec dataclass
# ══════════════════════════════════════════════════════════════════════

@dataclass
class StageSpec:
    """One curriculum step.  Model and loss are agnostic — they only see
    the flags this spec carries."""
    name:    str
    epochs:  int
    lr:      float
    # Trainable flags (model.set_trainable)
    encoder:        bool = True
    planner:        bool = True
    executor:       bool = True
    deform_only:    bool = False        # overrides ``executor``
    # Forward flags (model.forward)
    enable_physics: bool = False
    run_planner:    bool = True         # False → skip Planner forward (~10-15% faster)
    # Scheduled-sampling for AR decoder (PDF Stage-2: 温度/采样率扫参)
    # Linear ramp from 0 → sample_prob_max over sample_prob_ramp_epochs (within stage).
    # Default 0 = pure teacher forcing (early stages don't need it).
    sample_prob_max:         float = 0.0
    sample_prob_ramp_epochs: int   = 0
    # Render density for rec / lpips / depth losses (gsplat).  How many uniformly
    # spaced timesteps to rasterize per training step:
    #   0      → skip rendering entirely (rec losses early-exit to 0)
    #   2      → just initial + final (cheap, learns endpoints)
    #   5      → start + 3 mid + end (balanced default)
    #   T+1    → dense render (matches PDF spec, ~5× slower)
    # Set 0 in stages where no rendering-affected module is trainable (e.g.
    # PLANNER stage trains only Planner — wasted compute to render).
    render_n_timesteps: int = 5
    # Loss flags (CAPLoss.forward)
    loss:           LossSpec = field(default_factory=LossSpec)


# ══════════════════════════════════════════════════════════════════════
# Stage presets — edit these to change the curriculum
# ══════════════════════════════════════════════════════════════════════

DEFAULT_STAGES: List[StageSpec] = [
    StageSpec(
        name="RIGID", epochs=35, lr=3e-4,                # PDF: initial lr=3e-4
        encoder=True, planner=False, executor=True, deform_only=False,
        enable_physics=False,
        run_planner=True,            # PDF Stage-0: NCE warmup needs task_emb
                                     # (Planner is frozen so weights don't move,
                                     #  but text→task_emb path is computed for NCE)
        render_n_timesteps=5,        # encoder + executor train → rec_loss helps
        loss=LossSpec(),             # only always-on terms (clos/inv/comm/NCE/rec/VQ)
    ),
    StageSpec(
        name="PLANNER", epochs=35, lr=2e-4,              # ~1.5× decay from RIGID
        encoder=False, planner=True, executor=False, deform_only=False,
        enable_physics=False,
        run_planner=True,
        render_n_timesteps=0,        # only Planner trains — rendering gives no
                                     # gradient to any trainable module → skip
        loss=LossSpec(
            anneal_cvae_kl=True,     # β 0.01 → 0.1 ramp (anti posterior collapse)
            anneal_comm=True,        # L_comm 0.01 → 0.1 easy-to-hard ramp
        ),
    ),
    StageSpec(
        name="PHYSICS", epochs=25, lr=1e-4,              # ~2× decay from PLANNER
        encoder=False, planner=False, executor=False, deform_only=True,
        enable_physics=True,
        run_planner=False,           # planner is frozen here AND no NCE/hier/CVAE
                                     # losses are enabled → save ~10-15% per step
                                     # by skipping the Planner forward entirely.
        render_n_timesteps=5,        # executor.deform trains → rec_loss helps
        loss=LossSpec(
            enable_physics=True,
            enable_lipschitz=True,
            enable_physics_loss=True,
        ),
    ),
    StageSpec(
        name="FULL", epochs=55, lr=3e-5,                 # ~3× decay from PHYSICS
        encoder=True, planner=True, executor=True, deform_only=False,
        enable_physics=True,
        sample_prob_max=0.5,                             # PDF Stage-2: 采样率扫参
        sample_prob_ramp_epochs=15,                      # ramp 0→0.5 over first 15 of 55 ep
        render_n_timesteps=5,                            # full e2e — keep rec signal
        loss=LossSpec(
            enable_physics=True,
            enable_lipschitz=True,
            enable_physics_loss=True,
            enable_equiv=True,
            enable_hier=True,
            enable_entropy=True,
            anneal_cvae_kl=True,
            anneal_comm=True,
        ),
    ),
]


# Local pipeline smoke-test: 1 epoch per stage, all other flags inherited.
SMOKE_STAGES: List[StageSpec] = [
    StageSpec(name=s.name, epochs=1, lr=s.lr,
              encoder=s.encoder, planner=s.planner, executor=s.executor,
              deform_only=s.deform_only, enable_physics=s.enable_physics,
              run_planner=s.run_planner,
              sample_prob_max=s.sample_prob_max,
              sample_prob_ramp_epochs=s.sample_prob_ramp_epochs,
              render_n_timesteps=s.render_n_timesteps,
              loss=copy.deepcopy(s.loss))   # defensive copy — avoid SMOKE/DEFAULT alias
    for s in DEFAULT_STAGES
]


__all__ = ["StageSpec", "DEFAULT_STAGES", "SMOKE_STAGES"]
