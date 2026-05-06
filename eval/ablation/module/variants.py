"""variants.py — single source of truth for module-level ablations.

Each entry maps a variant *name* to:
  config_overrides  : nested dict of yaml keys to override in configs/config.yaml
  loss_overrides    : nested dict of yaml keys to override in configs/loss.yaml
                      (and configs/loss_b.yaml for the B fine-tune)
  trainer_flags     : list of CLI flags appended to torchrun (e.g. ['--no-physics'])
  description       : human-readable, used for log banners and aggregate tables

Conventions:
  - Path keys use dot-notation (e.g. ``"loss.lambda_clos"``); ``make_config.py``
    walks the dict on dump to produce the patched yaml.
  - These ablations are applied during BOTH the A train and the B fine-tune,
    so a single override propagates correctly across the full A→B pipeline.
  - The "main" model (no ablation) is the existing ``runs/main_exp/seed_X/``
    — no entry here.

PDF mapping:  §5.2 实验五 / 消融实验
Act4D MD §3.A 模块级
Experiment.md Tier 1 §2.B (rows 6-11)
"""

from __future__ import annotations

from typing import Any, Dict, List

# A loss term is "off" if its weight is set to exactly 0.  We keep the term
# in the config (rather than deleting the key) so that ``yaml.safe_load`` in
# the loss module still finds the entry — avoids KeyError lookups.

VARIANTS: Dict[str, Dict[str, Any]] = {

    # ──────────────────────────────────────────────────────────────────
    # 1.  w/o Hierarchical
    # ──────────────────────────────────────────────────────────────────
    "no_hier": {
        "description":
            "w/o Hierarchical: Planner uses LanguageEncoder text_emb directly "
            "(no TaskTokenizer / task codebook), so atomic tokens are conditioned "
            "on raw text rather than on hierarchical task tokens. Tests whether "
            "the J-token bottleneck contributes beyond raw text conditioning.",
        "config_overrides": {
            "planner.use_task_token": False,
        },
        "loss_overrides": {
            # Hierarchical consistency loss is undefined without task tokens.
            "loss.lambda_hier": 0.0,
        },
        "trainer_flags": [],
    },

    # ──────────────────────────────────────────────────────────────────
    # 2.  w/o Algebraic
    # ──────────────────────────────────────────────────────────────────
    "no_algebraic": {
        "description":
            "w/o Algebraic: zero out closure / inverse / equivariance / "
            "commutator weights. Atomic tokens still trained via VQ + rec + "
            "NCE, but no group-theoretic structure imposed. This is the "
            "central ablation — defines the paper's core contribution.",
        "config_overrides": {},
        "loss_overrides": {
            "loss.lambda_clos":     0.0,
            "loss.lambda_inv":      0.0,
            "loss.lambda_eq":       0.0,
            "loss.lambda_eq_cross": 0.0,
            "loss.lambda_comm":     0.0,
            "loss.lambda_comm_max": 0.0,
        },
        "trainer_flags": [],
    },

    # ──────────────────────────────────────────────────────────────────
    # 3.  w/o CVAE  (deterministic Planner)
    # ──────────────────────────────────────────────────────────────────
    "no_cvae": {
        "description":
            "w/o CVAE: deterministic Planner — disable KL term and recon term "
            "and force greedy decoding, so the latent z collapses to a point "
            "and Planner becomes a deterministic seq2seq. Tests whether sampled "
            "latent diversity is needed for action variety.",
        "config_overrides": {
            "planner.sampling_cfg.deterministic": True,
        },
        "loss_overrides": {
            "loss.lambda_cvae_kl":     0.0,
            "loss.lambda_cvae_kl_max": 0.0,
            "loss.lambda_cvae_recon":  0.0,
        },
        "trainer_flags": [],
    },

    # ──────────────────────────────────────────────────────────────────
    # 4.  w/o Physics  (use --no-physics CLI flag, see trainer.py)
    # ──────────────────────────────────────────────────────────────────
    "no_physics": {
        "description":
            "w/o Physics: enable_physics=False on every stage; the deformation "
            "branch falls back to identity (no MPM / PBD / Lipschitz vfield). "
            "Tests whether the differentiable physics plugin contributes "
            "beyond rigid SE(3) execution alone.",
        "config_overrides": {},
        "loss_overrides": {
            "loss.lambda_physics":   0.0,
            "loss.lambda_lip":       0.0,    # vfield isn't exercised → no Lip needed
        },
        "trainer_flags": ["--no-physics"],
    },

    # ──────────────────────────────────────────────────────────────────
    # 5.  w/o Equivariance
    # ──────────────────────────────────────────────────────────────────
    "no_equivariance": {
        "description":
            "w/o Equivariance: zero out L_eq and L_eq_cross; closure / inverse "
            "still active. Tests whether SE(3) equivariance specifically "
            "contributes (vs the broader algebraic structure).",
        "config_overrides": {},
        "loss_overrides": {
            "loss.lambda_eq":       0.0,
            "loss.lambda_eq_cross": 0.0,
        },
        "trainer_flags": [],
    },

    # ──────────────────────────────────────────────────────────────────
    # 6.  w/o Lipschitz
    # ──────────────────────────────────────────────────────────────────
    "no_lipschitz": {
        "description":
            "w/o Lipschitz: zero out L_Lip (no spectral-norm penalty on "
            "vfield Jacobian).  Note: spectral_norm WRAPPING in vfield is left "
            "intact — only the loss term is removed. Tests whether the "
            "explicit Lipschitz penalty is needed beyond architectural "
            "constraints.",
        "config_overrides": {},
        "loss_overrides": {
            "loss.lambda_lip": 0.0,
        },
        "trainer_flags": [],
    },
}


def list_variants() -> List[str]:
    return sorted(VARIANTS.keys())


def get_variant(name: str) -> Dict[str, Any]:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}; known: {list_variants()}")
    return VARIANTS[name]
