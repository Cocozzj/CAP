"""Manifest entry → PDDL problem (and back: plan → action sequence).

Bridges our dataset's symbolic task descriptions to PDDLStream's expected
problem specification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# Verb decomposition
# ──────────────────────────────────────────────────────────────────────

# Composite tasks ("comp:close_open") expand to atomic verb sequences.
def decompose_task(task_name: str) -> List[str]:
    """Expand a manifest task_name into a list of atomic PDDL action names.

    Examples:
      "open"                       → ["open"]
      "close"                      → ["close"]
      "comp:close_open"            → ["close", "open"]
      "comp:open_close_open_close" → ["open", "close", "open", "close"]
      "comp:open_open_more"        → ["open", "open"]    ("more" is dropped)
      "fold"                       → ["fold"]            (TAMP can't execute, but symbolic OK)
    """
    if not task_name:
        return []
    if not task_name.startswith("comp:"):
        return [task_name]
    parts = task_name[len("comp:"):].split("_")
    return [p for p in parts if p and p != "more"]


# ──────────────────────────────────────────────────────────────────────
# Build PDDLStream problem
# ──────────────────────────────────────────────────────────────────────

def build_problem(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Translate one trajectory's meta.json into a PDDL problem.

    Returns a dict shaped like:
      {
        "objects": [(name, type), ...],
        "init":    [(predicate, *args), ...],
        "goal":    (predicate, *args)  OR  ("and", goal_clauses...)
        "atomic_plan": [verb1, verb2, ...]   <- decomposition of task_name
      }

    Returns None if the task is not in PDDL domain.
    """
    task_name = meta.get("task_name", "")
    obj_id    = str(meta.get("obj_id", "obj0"))
    obj_cat   = meta.get("obj_category", "")
    joint_idx = int(meta.get("joint_index", 0))
    joint_name= str(meta.get("joint_name", "joint0"))

    plan = decompose_task(task_name)
    if not plan:
        return None

    # Build a stable object name we can refer to in PDDL
    obj_pddl  = f"o_{obj_cat}_{obj_id}".replace(":", "_").replace("-", "_")
    j_pddl    = f"j_{joint_name}".replace(":", "_").replace("-", "_")

    # Initial facts (start state)
    init: List[Tuple] = [
        ("Articulated", obj_pddl),
        ("HasJoint",    obj_pddl, j_pddl),
        ("Revolute",    j_pddl),                    # default; mostly true for PartNet
        ("JointAngle",  j_pddl, "a-cur"),
        ("JointMin",    j_pddl, "a-min"),
        ("JointMax",    j_pddl, "a-max"),
    ]
    objects: List[Tuple[str, str]] = [
        (obj_pddl, "object"),
        (j_pddl,   "joint"),
        ("a-cur",  "angle"),
        ("a-min",  "angle"),
        ("a-max",  "angle"),
    ]

    # If task starts from a non-default state (e.g. "close" from a half-open
    # box), randomization tells us start_fraction.
    rand = meta.get("randomization", {}) or {}
    start_frac  = float(rand.get("start_fraction",  0.0))
    target_frac = float(rand.get("target_fraction", 1.0))

    # Initial Open/Closed depending on start_fraction
    if start_frac > 0.5:
        init.append(("Open", obj_pddl))
    else:
        init.append(("Closed", obj_pddl))

    # Goal: derived from the LAST atomic action in the plan
    # (compositional plans like "close_open" end at "open" → goal is Open)
    last = plan[-1]
    goal: Tuple
    if last == "open":      goal = ("Open",     obj_pddl)
    elif last == "close":   goal = ("Closed",   obj_pddl)
    elif last == "rotate":  goal = ("Rotated",  obj_pddl)
    elif last == "push":    goal = ("Pushed",   obj_pddl)
    elif last == "pull":    goal = ("Pulled",   obj_pddl)
    elif last == "fold":    goal = ("Folded",   obj_pddl)
    elif last == "squeeze": goal = ("Squeezed", obj_pddl)
    elif last == "pour":    goal = ("Poured",   obj_pddl)
    else:
        # Unknown verb
        return None

    # For push/pull we also need a starting position
    if any(v in plan for v in ("push", "pull")):
        objects.extend([("p1", "pose"), ("p2", "pose")])
        init.append(("At", obj_pddl, "p1"))

    return {
        "objects":     objects,
        "init":        init,
        "goal":        goal,
        "atomic_plan": plan,
        "obj_pddl":    obj_pddl,
        "j_pddl":      j_pddl,
        "metadata":    {
            "obj_id":           obj_id,
            "obj_category":     obj_cat,
            "task_name":        task_name,
            "start_fraction":   start_frac,
            "target_fraction":  target_frac,
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Pretty-print PDDL problem (for debugging or feeding to Fast-Downward)
# ──────────────────────────────────────────────────────────────────────

def to_pddl_problem_str(problem: Dict[str, Any], domain_name: str = "partnet-mobility-tamp") -> str:
    """Render the dict as a PDDL problem definition string."""
    name = f"problem-{problem['obj_pddl']}"
    obj_lines = []
    for obj_name, obj_type in problem["objects"]:
        obj_lines.append(f"    {obj_name} - {obj_type}")
    init_lines = []
    for fact in problem["init"]:
        init_lines.append("    (" + " ".join(str(a) for a in fact) + ")")
    goal = problem["goal"]
    goal_str = "(" + " ".join(str(a) for a in goal) + ")"
    return f"""(define (problem {name})
  (:domain {domain_name})
  (:objects
{chr(10).join(obj_lines)}
  )
  (:init
{chr(10).join(init_lines)}
  )
  (:goal {goal_str})
)
"""
