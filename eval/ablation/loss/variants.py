"""variants.py — Tier 2 loss-term ablations.

Each variant zero-outs ONE loss weight to isolate that term's contribution.
A-only training (per Experiment.md Tier 2 budget cap); not retrained on B.

Format identical to module/variants.py — the same make_config.py + train_a.sh
infrastructure works on both.
"""

from __future__ import annotations

from typing import Any, Dict, List

VARIANTS: Dict[str, Dict[str, Any]] = {

    "no_L_clos": {
        "description":
            "Drop closure loss alone (Theorem 1: closure error bound). "
            "Tests whether L_clos is the term that empirically enforces the "
            "K^{-1/d} closure-error scaling — closure_gap should rise sharply "
            "while inverse_gap and other metrics stay closer to baseline.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_clos": 0.0},
        "trainer_flags":    [],
    },

    "no_L_inv": {
        "description":
            "Drop inverse loss alone (Theorem 2: inverse consistency bound). "
            "Closure / equivariance still pull toward group structure; "
            "isolates the unique contribution of L_inv to the inverse_gap.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_inv": 0.0},
        "trainer_flags":    [],
    },

    "no_L_eq": {
        "description":
            "Drop equivariance loss alone (Proposition 3: cross-object "
            "equivariance bound).  Closure / inverse / commutator still "
            "active; tests whether SE(3)-equivariance training is the "
            "specific term that drives cross-object transfer accuracy.",
        "config_overrides": {},
        "loss_overrides":   {
            "loss.lambda_eq":       0.0,
            "loss.lambda_eq_cross": 0.0,
        },
        "trainer_flags":    [],
    },

    "no_L_comm": {
        "description":
            "Drop commutator regulariser (and its anneal). Useful since L_comm "
            "is a soft probe rather than a strict requirement — paper claims "
            "L_comm is auxiliary, this run defends that claim quantitatively.",
        "config_overrides": {},
        "loss_overrides":   {
            "loss.lambda_comm":     0.0,
            "loss.lambda_comm_max": 0.0,
        },
        "trainer_flags":    [],
    },

    "no_L_hier": {
        "description":
            "Drop hierarchical consistency loss (Proposition 4: hierarchical "
            "algebraic error bound).  Task-tokens still exist but no explicit "
            "penalty enforces task↔atomic alignment beyond CVAE training. "
            "Tests whether L_hier is the specific term that bounds the "
            "decomposition error between task and atomic levels.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_hier": 0.0},
        "trainer_flags":    [],
    },

    "no_L_nce": {
        "description":
            "Drop InfoNCE alignment of task ↔ text (Proposition 5: semantic "
            "coherence).  Task token still trained via VQ + CVAE recon, but "
            "no contrastive signal to text labels.  Tests whether removing "
            "the lower-bound on I(c_task; verb) breaks text-conditional "
            "generation correctness.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_nce": 0.0},
        "trainer_flags":    [],
    },

    "no_kl_anneal": {
        "description":
            "Disable CVAE KL beta annealing (β stays at the fixed yaml value "
            "from step 0 instead of 0.01→0.1 ramp). Tests whether the "
            "anti-posterior-collapse schedule is needed.",
        "config_overrides": {},
        "loss_overrides":   {},
        "trainer_flags":    ["--no-kl-anneal"],
    },
}


def list_variants() -> List[str]:
    return sorted(VARIANTS.keys())


def get_variant(name: str) -> Dict[str, Any]:
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant {name!r}; known: {list_variants()}")
    return VARIANTS[name]
