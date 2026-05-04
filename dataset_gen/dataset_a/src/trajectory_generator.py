"""Generate the full set of trajectories.

Atomic trajectories: every (object × applicable atomic task × seed).
Composition trajectories (optional): every (object × applicable composition × seed),
both 2-step (in train) and 3-step+ (eval-only).

Reads:
  - object_list.json         (output of Step 1)
  - configs/tasks.yaml
  - configs/compositions.yaml (optional)
  - configs/default.yaml

Writes:
  - trajectories.json (atomic + composition together; renderer reads this)
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .tasks import ALL_TASK_CLASSES, CompositeTask, get_task_class

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# applicability + id helpers
# ----------------------------------------------------------------------------
def applicable_tasks_for_category(task_cfg: dict, category: str) -> List[dict]:
    out = []
    for t in task_cfg["tasks"]:
        if category in t.get("applies_to", []):
            out.append(t)
    return out


def applicable_compositions_for_category(comp_list: list, category: str) -> List[dict]:
    return [c for c in comp_list if category in c.get("applies_to", [])]


def make_atomic_traj_id(obj_record: dict, task_name: str, seed: int) -> str:
    return f"A_{obj_record['our_category']}_{obj_record['obj_id']}_{task_name}_s{seed:03d}"


def make_composite_traj_id(obj_record: dict, comp_name: str, seed: int) -> str:
    return f"A_{obj_record['our_category']}_{obj_record['obj_id']}_comp-{comp_name}_s{seed:03d}"


# ----------------------------------------------------------------------------
# top-level generator
# ----------------------------------------------------------------------------
def generate_all_trajectories(
    object_list: List[dict],
    tasks_cfg: dict,
    physics_cfg: dict,
    *,
    trajectories_per_pair: int,
    target_total: int,
    fps: int = 30,
    seed: int = 42,
    compositions_cfg: Optional[dict] = None,
    num_workers: int = 1,
    verbose: bool = True,
) -> List[dict]:
    """Generate atomic + (optional) composition trajectories.

    Atomic trajectories follow the original `target_total` budget.
    Composition trajectories are *additional* (not counted against `target_total`).
    """
    import sapien.core as sapien

    rng_top = random.Random(seed)

    # ---- prepare configs
    atomic_defaults_cfg = tasks_cfg.get("defaults", {
        "total_duration_seconds": 13.0,
        "pre_settle_range": [1.5, 3.0],
        "post_settle_range": [5.0, 8.0],
    })
    physics_with_fps = dict(physics_cfg)
    physics_with_fps["fps"] = fps

    atomic_task_cfgs_by_name = {t["name"]: t for t in tasks_cfg["tasks"]}

    # ---- build atomic plan
    atomic_plan: List[tuple[dict, dict, int]] = []
    for obj in object_list:
        cat = obj["our_category"]
        for tcfg in applicable_tasks_for_category(tasks_cfg, cat):
            for k in range(trajectories_per_pair):
                atomic_plan.append((obj, tcfg, rng_top.randint(0, 2**31 - 1)))

    if verbose:
        logger.info("Atomic plan: %d trajectories", len(atomic_plan))

    rng_top.shuffle(atomic_plan)
    if len(atomic_plan) > target_total:
        atomic_plan = _stratified_subsample(atomic_plan, target_total, rng_top)
        if verbose:
            logger.info("Atomic subsampled to %d", len(atomic_plan))

    # ---- build composition plans (2-step + eval-only)
    comp_plan_in_train: List[tuple[dict, dict, int]] = []
    comp_plan_eval_only: List[tuple[dict, dict, int]] = []
    if compositions_cfg:
        for comp in compositions_cfg.get("compositions_train_eval", []):
            for obj in object_list:
                if obj["our_category"] not in comp.get("applies_to", []):
                    continue
                # All sub-tasks must apply to this object
                if not _all_subtasks_applicable(comp, obj["our_category"], tasks_cfg):
                    continue
                for k in range(comp.get("trajectories_per_pair", 3)):
                    comp_plan_in_train.append(
                        (obj, comp, rng_top.randint(0, 2**31 - 1))
                    )
        for comp in compositions_cfg.get("compositions_eval_only", []):
            for obj in object_list:
                if obj["our_category"] not in comp.get("applies_to", []):
                    continue
                if not _all_subtasks_applicable(comp, obj["our_category"], tasks_cfg):
                    continue
                for k in range(comp.get("trajectories_per_pair", 1)):
                    comp_plan_eval_only.append(
                        (obj, comp, rng_top.randint(0, 2**31 - 1))
                    )

        if verbose:
            logger.info("Composition plans: %d in-train, %d eval-only",
                        len(comp_plan_in_train), len(comp_plan_eval_only))

    # ---- assemble unified work list
    # Each item: ("atomic" | "comp_train" | "comp_eval", obj, cfg_or_comp, seed)
    work_items: List[tuple] = []
    for obj, tcfg, sd in atomic_plan:
        work_items.append(("atomic", obj, tcfg, sd))
    if compositions_cfg:
        comp_defaults_cfg = compositions_cfg.get("defaults", {})
        for obj, comp, sd in comp_plan_in_train:
            work_items.append(("comp_train", obj, comp, sd))
        for obj, comp, sd in comp_plan_eval_only:
            work_items.append(("comp_eval", obj, comp, sd))
    else:
        comp_defaults_cfg = {}

    if verbose:
        logger.info("Total work items: %d (atomic=%d, comp_train=%d, comp_eval=%d)",
                    len(work_items), len(atomic_plan),
                    len(comp_plan_in_train) if compositions_cfg else 0,
                    len(comp_plan_eval_only) if compositions_cfg else 0)

    # ---- run (parallel via num_workers, or serial if num_workers <= 1)
    serial_args = (
        atomic_task_cfgs_by_name,
        atomic_defaults_cfg,
        comp_defaults_cfg,
        physics_with_fps,
        physics_cfg,
    )

    out_records: List[dict] = []
    if num_workers <= 1:
        out_records = _run_chunk(work_items, *serial_args, verbose=verbose)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Split work into chunks (one per worker, balanced)
        chunks: List[List[tuple]] = [[] for _ in range(num_workers)]
        for i, item in enumerate(work_items):
            chunks[i % num_workers].append(item)
        if verbose:
            logger.info("Dispatching to %d workers (avg %.0f items each)",
                        num_workers, len(work_items) / num_workers)
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [ex.submit(_run_chunk, chunk, *serial_args, verbose=False)
                       for chunk in chunks]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Workers", disable=not verbose):
                try:
                    out_records.extend(fut.result())
                except Exception as e:  # noqa: BLE001
                    logger.exception("Worker failed: %s", e)

    if verbose:
        n_success = sum(1 for r in out_records if r["success"])
        n_atomic = sum(1 for r in out_records if not r.get("is_composition"))
        n_comp_train = sum(1 for r in out_records
                           if r.get("is_composition") and not r.get("eval_only"))
        n_comp_eval = sum(1 for r in out_records
                          if r.get("is_composition") and r.get("eval_only"))
        logger.info(
            "Generated %d total (%d successful) | atomic=%d, comp-train=%d, comp-eval-only=%d",
            len(out_records), n_success, n_atomic, n_comp_train, n_comp_eval,
        )

    return out_records


# ----------------------------------------------------------------------------
# worker: process a chunk of work items in a single process
# ----------------------------------------------------------------------------
def _run_chunk(
    work_items: List[tuple],
    atomic_task_cfgs_by_name: dict,
    atomic_defaults_cfg: dict,
    comp_defaults_cfg: dict,
    physics_with_fps: dict,
    physics_cfg: dict,
    *,
    verbose: bool = False,
) -> List[dict]:
    """Process a chunk of work items. Designed for ProcessPoolExecutor:
    each worker process creates its own SAPIEN engine + renderer.
    """
    import sapien.core as sapien

    engine = sapien.Engine()
    sap_renderer = sapien.SapienRenderer()
    engine.set_renderer(sap_renderer)

    out: List[dict] = []
    iterator = work_items
    if verbose:
        iterator = tqdm(work_items, desc="Trajectories")

    for kind, obj, cfg_or_comp, sd in iterator:
        try:
            if kind == "atomic":
                tcfg = cfg_or_comp
                TaskCls = get_task_class(tcfg["name"])
                task = TaskCls(tcfg, atomic_defaults_cfg, physics_with_fps)
                traj_id = make_atomic_traj_id(obj, tcfg["name"], sd)

                is_soft_task = tcfg["name"] in ("squeeze", "fold")
                if is_soft_task:
                    rec = task.generate(obj, seed=sd, traj_id=traj_id, scene=None)
                else:
                    scene = engine.create_scene()
                    scene.set_timestep(physics_cfg.get("step_dt", 0.0033))
                    scene.add_ground(altitude=-1.0)
                    rec = task.generate(obj, seed=sd, traj_id=traj_id, scene=scene)
                    del scene
            else:
                # composite (train or eval-only)
                comp = cfg_or_comp
                eval_only = (kind == "comp_eval")
                scene = engine.create_scene()
                scene.set_timestep(physics_cfg.get("step_dt", 0.0033))
                scene.add_ground(altitude=-1.0)
                ctask = CompositeTask(
                    comp_cfg=comp,
                    atomic_task_cfgs=atomic_task_cfgs_by_name,
                    atomic_task_classes=ALL_TASK_CLASSES,
                    atomic_defaults_cfg=atomic_defaults_cfg,
                    comp_defaults_cfg=comp_defaults_cfg,
                    physics_cfg=physics_with_fps,
                    eval_only=eval_only,
                )
                traj_id = make_composite_traj_id(obj, comp["name"], sd)
                rec = ctask.generate(obj, seed=sd, traj_id=traj_id, scene=scene)
                del scene

            if rec is not None:
                out.append(rec.to_dict())
        except Exception as e:  # noqa: BLE001
            logger.warning("Item failed (%s, obj=%s): %s",
                           kind, obj.get("obj_id"), e)
            continue
    return out


def _all_subtasks_applicable(comp_cfg: dict, category: str, tasks_cfg: dict) -> bool:
    """Check that EVERY sub-task in comp.base_tasks applies to this category."""
    for sub_name in comp_cfg["base_tasks"]:
        sub_cfg = next((t for t in tasks_cfg["tasks"] if t["name"] == sub_name), None)
        if sub_cfg is None:
            return False
        if category not in sub_cfg.get("applies_to", []):
            return False
    return True


def _stratified_subsample(plan, target_n, rng):
    """Keep coverage of (category × task) pairs while shrinking to target_n."""
    buckets = defaultdict(list)
    for item in plan:
        cat = item[0]["our_category"]
        tn = item[1]["name"]
        buckets[(cat, tn)].append(item)

    out = []
    while len(out) < target_n and any(buckets.values()):
        for k in list(buckets.keys()):
            if not buckets[k]:
                continue
            out.append(buckets[k].pop())
            if len(out) >= target_n:
                break
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------------
class _NumpyJSONEncoder(json.JSONEncoder):
    """Handle numpy scalar / array types that show up in trajectory records."""

    def default(self, o):
        import numpy as np
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_trajectories(records: List[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trajectories": records, "n": len(records)},
                  f, indent=2, cls=_NumpyJSONEncoder)
    logger.info("Wrote %d trajectories to %s", len(records), out_path)


def load_trajectories(path: str | Path) -> List[dict]:
    with open(path) as f:
        return json.load(f)["trajectories"]
