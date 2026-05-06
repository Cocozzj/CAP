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
            "Drop closure loss alone (keep inverse / equiv / commutator). "
            "Tests whether closure error per se drives downstream metrics or "
            "if it's redundant with inverse.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_clos": 0.0},
        "trainer_flags":    [],
    },

    "no_L_inv": {
        "description":
            "Drop inverse loss alone. Closure / equivariance still pull toward "
            "group structure; isolates the unique contribution of L_inv.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_inv": 0.0},
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
            "Drop hierarchical consistency loss; task-tokens still exist but "
            "no explicit penalty enforces task↔atomic alignment beyond CVAE "
            "training. Justifies the extra L_hier term.",
        "config_overrides": {},
        "loss_overrides":   {"loss.lambda_hier": 0.0},
        "trainer_flags":    [],
    },

    "no_L_nce": {
        "description":
            "Drop InfoNCE alignment of task ↔ text. Task token still trained "
            "via VQ + CVAE recon, but no contrastive signal to text labels. "
            "Tests Proposition 5 (语法语义一致性).",
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
