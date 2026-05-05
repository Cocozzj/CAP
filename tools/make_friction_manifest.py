"""DEPRECATED — replaced by ``tools.make_topology_manifest``.

The Cross-friction column was replaced by Cross-Topology (rigid → soft) in the
paper's main table because friction in our SAPIEN-randomized data is a
simulator parameter, not a true material attribute.  See
``tools/make_topology_manifest.py`` for the topology-based replacement.

Safe to delete this file:
    rm tools/make_friction_manifest.py
"""
import sys

print(
    "ERROR: tools.make_friction_manifest is deprecated; "
    "use tools.make_topology_manifest instead.\n"
    "  python -m tools.make_topology_manifest \\\n"
    "      --manifest dataset/dataset_a/manifest.json \\\n"
    "      --output dataset/dataset_a/manifest_topology.json",
    file=sys.stderr,
)
sys.exit(1)
