"""TAMP via PDDLStream — classical Task and Motion Planning baseline.

Replaces the earlier ``tamp_rule`` (hand-coded if-then) with a real planner
based on:
  - PDDL 2.1 domain definition (predicates, actions, preconditions, effects)
  - PDDLStream library for combined task + motion planning
  - Standard motion primitives for object articulation (joint sweeps)

Paper positioning: "TAMP (PDDLStream)" in the main 5-baseline matrix.
Contrasts ``learned`` vs ``hand-written rules + classical search``.

Submodules:
  domain     PDDL domain file + python predicate generation
  motion     motion primitive library (open / close / push / pull)
  run_tamp   main entry — iterate split → write per-trajectory output
"""
