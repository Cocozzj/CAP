"""
Dataset-A → natural language verb phrases.

The Dataset-A manifest stores raw ``task_name`` strings (``"open"``,
``"close"``, ``"comp:open_close_open"``, ``"comp:open_open_more"`` ...) and
``obj_category`` (``"StorageFurniture_Drawer"``, ``"Door"`` ...).  The model's
LanguageEncoder wants natural language verb phrases.  This module is the
*only* place the dataset's naming schema leaks into the rest of the codebase.

If you swap to a different dataset whose ``task_name`` is already a natural
sentence, just stop calling ``task_to_text`` in ``dataloader.py``.
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────
# Verb templates per atomic task
# ────────────────────────────────────────────────────────────────────
_VERB_TEMPLATES = {
    "open":    "open the {cat}",
    "close":   "close the {cat}",
    "pull":    "pull the {cat}",
    "push":    "push the {cat}",
    "rotate":  "rotate the {cat}",
    "squeeze": "squeeze the {cat}",
    "fold":    "fold the {cat}",
    "pour":    "pour from the {cat}",
}

# ────────────────────────────────────────────────────────────────────
# obj_category → readable noun (for {cat} substitution)
# ────────────────────────────────────────────────────────────────────
_CATEGORY_NL = {
    "Box":                     "box",
    "Cloth":                   "cloth",
    "Dishwasher":              "dishwasher",
    "Door":                    "door",
    "Faucet":                  "faucet",
    "Kettle":                  "kettle",
    "Laptop":                  "laptop",
    "Microwave":               "microwave",
    "Oven":                    "oven",
    "Refrigerator":            "refrigerator",
    "Scissors":                "scissors",
    "SoftToy":                 "soft toy",
    "StorageFurniture_Door":   "cabinet door",
    "StorageFurniture_Drawer": "drawer",
    "Suitcase":                "suitcase",
    "TrashCan":                "trash can",
    "Window":                  "window",
}


def task_to_text(task_name: str, obj_category: str) -> str:
    """Build a natural-language verb phrase from task + category.

    Atomic:        ``"open"``      + ``"StorageFurniture_Drawer"`` → ``"open the drawer"``
    2-step comp:   ``"comp:open_close"``        + ``"Door"`` → ``"open then close the door"``
    3-step comp:   ``"comp:open_close_open"``   + ``"Door"`` → ``"open, close, then open the door"``
    Special:       ``"comp:open_open_more"``               → ``"open, then open the X further"``
    """
    cat_nl = _CATEGORY_NL.get(obj_category, obj_category.lower())

    # Atomic case
    if not task_name.startswith("comp:"):
        tpl = _VERB_TEMPLATES.get(task_name, task_name + " the {cat}")
        return tpl.format(cat=cat_nl)

    # Composite case
    steps = task_name[len("comp:"):].split("_")

    # Special: "..._more" suffix → add "further"
    if steps[-1] == "more" and len(steps) >= 2:
        head = steps[:-1]
        if len(head) == 1:
            return f"{head[0]} the {cat_nl} further"
        joined = ", ".join(head[:-1]) + f", then {head[-1]}"
        return f"{joined} the {cat_nl} further"

    if len(steps) == 1:
        return _VERB_TEMPLATES.get(steps[0], steps[0]).format(cat=cat_nl)
    if len(steps) == 2:
        return f"{steps[0]} then {steps[1]} the {cat_nl}"
    head = ", ".join(steps[:-1])
    return f"{head}, then {steps[-1]} the {cat_nl}"


__all__ = ["task_to_text"]
