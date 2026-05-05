"""Per-dataset text formatting.

DatasetA's manifest stores raw ``task_name`` strings ("open", "close",
"comp:open_close", ...) and ``obj_category`` ("Door", ...).  The model's
LanguageEncoder wants natural-language verb phrases — ``task_to_text`` is
the only place that schema leaks into the codebase.

DatasetB (Something Something v2) entries already carry natural English in
``raw_label`` ("closing bucket"); ``dataset_b_text`` just plucks it (with
template + placeholders fallback).
"""

from __future__ import annotations

from typing import Dict


# ════════════════════════════════════════════════════════════════════
# DatasetA — composite task_name + obj_category → natural verb phrase
# ════════════════════════════════════════════════════════════════════

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

    Atomic:        "open"      + "StorageFurniture_Drawer" → "open the drawer"
    2-step comp:   "comp:open_close"     + "Door" → "open then close the door"
    3-step comp:   "comp:open_close_open" + "Door" → "open, close, then open the door"
    Special:       "comp:open_open_more"           → "open, then open the X further"
    """
    cat_nl = _CATEGORY_NL.get(obj_category, obj_category.lower())

    if not task_name.startswith("comp:"):
        tpl = _VERB_TEMPLATES.get(task_name, task_name + " the {cat}")
        return tpl.format(cat=cat_nl)

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


# ════════════════════════════════════════════════════════════════════
# DatasetB — Something Something v2 has natural English labels already
# ════════════════════════════════════════════════════════════════════

import re

# SSv2 template placeholders.  In published SSv2 templates the variants seen
# in the wild are: [something], [something else], [something in something],
# [number of] (rare).  We match any "[ ... ]" group and fill them in order
# from ``placeholders`` so we never leave dangling brackets in the output.
_SSV2_PLACEHOLDER = re.compile(r"\[[^\[\]]+\]")


def dataset_b_text(entry: Dict) -> str:
    """Extract natural-language label from a DatasetB manifest entry.

    Priority:
      1. ``raw_label``                          ("closing bucket")
      2. ``template`` filled IN ORDER with ``placeholders``:
            "Putting [something] on [something]" + ["mug", "table"]
              → "Putting mug on table"
         Handles all SSv2 placeholder variants ([something], [something else],
         [something in something]) generically via regex — replaces every
         "[...]" group sequentially with the next placeholder.
      3. ``task_name + obj_category`` fallback so it never crashes
    """
    raw = entry.get("raw_label")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()

    tpl = entry.get("template")
    phs = entry.get("placeholders") or []
    if isinstance(tpl, str) and tpl.strip():
        it = iter(phs)

        def _sub(_match):
            try:
                return str(next(it))
            except StopIteration:
                return ""        # leave nothing if we run out of placeholders
        text = _SSV2_PLACEHOLDER.sub(_sub, tpl)
        # Collapse double spaces left by empty-placeholder substitutions
        return re.sub(r"\s+", " ", text).strip()

    return f"{entry.get('task_name', 'do')} {entry.get('obj_category', '')}".strip()


__all__ = ["task_to_text", "dataset_b_text"]
