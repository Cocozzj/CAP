"""DEPRECATED — replaced by ``tools.inspect_topology``.

The Cross-friction column was replaced by Cross-Topology (rigid → soft) in the
paper's main table because friction in our SAPIEN-randomized data is a
simulator parameter, not a true material attribute.  See
``tools/inspect_topology.py`` for the topology-based replacement.

Safe to delete this file:
    rm tools/inspect_friction.py
"""
import sys

print(
    "ERROR: tools.inspect_friction is deprecated; use tools.inspect_topology instead.\n"
    "  python -m tools.inspect_topology --manifest dataset/dataset_a/manifest.json",
    file=sys.stderr,
)
sys.exit(1)
