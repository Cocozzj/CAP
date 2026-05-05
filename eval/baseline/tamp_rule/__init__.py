"""DEPRECATED — replaced by ``eval.baseline.tamp_pddl``.

The hand-coded rule baseline was upgraded to full PDDLStream-based TAMP
(Task and Motion Planning) for the paper's main table.

Safe to delete this directory:
    rm -rf eval/baseline/tamp_rule
"""
import warnings
warnings.warn(
    "eval.baseline.tamp_rule is deprecated; use eval.baseline.tamp_pddl instead",
    DeprecationWarning, stacklevel=2,
)
