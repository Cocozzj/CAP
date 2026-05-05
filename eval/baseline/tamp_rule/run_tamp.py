"""DEPRECATED — moved to eval.baseline.tamp_pddl.run_tamp."""
import sys
print(
    "ERROR: eval.baseline.tamp_rule.run_tamp was renamed to "
    "eval.baseline.tamp_pddl.run_tamp.\n"
    "Run:\n"
    "  python -m eval.baseline.tamp_pddl.run_tamp ...",
    file=sys.stderr,
)
sys.exit(1)
