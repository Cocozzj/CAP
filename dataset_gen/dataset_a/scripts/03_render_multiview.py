"""Step 3: Render each trajectory from N cameras and write MP4 videos.

Reads:  trajectories.json + cameras.yaml + default.yaml
Writes: <out>/<traj_id>/{front,side,...}.mp4 + meta + physics + cameras.json

GPU strategy:
    Each worker process is pinned to ONE GPU via CUDA_VISIBLE_DEVICES,
    set BEFORE sapien is imported. SAPIEN's Vulkan renderer follows
    CUDA_VISIBLE_DEVICES to pick the GPU. Worker→GPU mapping is
    `pid % n_gpu`, which gives an even spread.

    Implementation: we use multiprocessing's default "fork" start method
    (works on Linux and inherits the parent's site-packages / venv). For
    GPU pinning to work, the PARENT must NOT have imported sapien yet —
    otherwise the Vulkan/CUDA context is already initialised when workers
    are forked, and CUDA_VISIBLE_DEVICES set in the worker has no effect.

    To keep the parent sapien-free, this script avoids any top-level
    import of sapien-using modules. In particular:
      * `src.trajectory_generator` imports sapien at module top — we do
        NOT import it; instead we inline the JSON loading.
      * `src.camera_setup` only imports sapien inside functions, so
        `parse_cameras_yaml` is safe at module-load.
      * `src.renderer` is imported lazily inside `_worker` (in the child),
        AFTER `_init_worker` has set CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml  # noqa: E402  — only YAML at top, no sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _load_trajectories(path):
    """Inline JSON loader so the parent process does NOT import
    src.trajectory_generator (which imports sapien at module top)."""
    with open(path) as f:
        d = json.load(f)
    return d["trajectories"] if isinstance(d, dict) and "trajectories" in d else d


# ----------------------------------------------------------------------
# worker
# ----------------------------------------------------------------------
def _init_worker(n_gpu: int):
    """Run ONCE per worker process, BEFORE any sapien / pyrender import.

    Pins this worker to a single GPU via CUDA_VISIBLE_DEVICES (which
    SAPIEN's Vulkan renderer will follow). pid % n_gpu gives an even
    spread without needing a manager queue.

    Also forces pyrender (used by src.soft_renderer for cloth / soft-toy /
    pour trajectories) onto the headless EGL backend. Without this, pyrender
    falls back to pyglet → X11 → NoSuchDisplayException on a headless server.
    Setting PYOPENGL_PLATFORM here (before any rendering library is imported
    in the worker) is reliable; relying on src.soft_renderer's own setdefault
    is not, because pyrender can be imported transitively before that module
    runs.
    """
    pid = os.getpid()
    gpu_id = pid % max(n_gpu, 1)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Some SAPIEN builds also honor this; harmless if ignored.
    os.environ.setdefault("SAPIEN_RENDER_DEVICE", f"cuda:{gpu_id}")
    # Headless EGL for pyrender (soft trajectories: cloth, soft toy, pour).
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    print(f"[pid={pid}] CUDA_VISIBLE_DEVICES={gpu_id} "
          f"PYOPENGL_PLATFORM={os.environ['PYOPENGL_PLATFORM']}", flush=True)


def _worker(args_tuple):
    """Render one trajectory. sapien-using modules are imported HERE
    (lazy), AFTER _init_worker has set CUDA_VISIBLE_DEVICES."""
    traj, obj_record, cam_specs, camera_design, cfg, out_dir = args_tuple
    out_path = Path(out_dir) / traj["traj_id"]
    if out_path.exists() and (out_path / "meta.json").exists():
        # Match the 3-tuple shape returned in the rendering path so the
        # caller can always do `tid, status, dt = fut.result()`.
        return traj["traj_id"], "skipped", 0.0

    # Lazy import — sapien comes in only after CUDA_VISIBLE_DEVICES is set.
    from src.renderer import render_trajectory

    t0 = time.time()
    try:
        res = render_trajectory(
            traj_record=traj,
            obj_record=obj_record,
            camera_specs=cam_specs,
            image_size=cfg["render"]["resolution"],
            save_depth=cfg["render"].get("save_depth", False),
            out_dir=out_path,
            fps=cfg["render"]["fps"],
            camera_design=camera_design,
        )
        status = "ok" if res else "failed"
    except Exception as e:  # noqa: BLE001
        status = f"error: {type(e).__name__}: {e}"
    dt = time.time() - t0
    return traj["traj_id"], status, dt


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--object_list", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cameras", default="configs/cameras.yaml")
    ap.add_argument("--out", default="outputs/data/")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="Number of parallel worker processes")
    ap.add_argument("--n_gpu", type=int, default=4,
                    help="Number of GPUs available; workers spread "
                         "via pid %% n_gpu. Set to 1 if single-GPU.")
    ap.add_argument("--only_success", action="store_true",
                    help="Skip trajectories with success=False")
    ap.add_argument("--limit", type=int, default=None,
                    help="Render only first N trajectories (smoke test)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # camera_setup.parse_cameras_yaml only does lazy sapien imports
    # (sapien is imported inside its functions, not at module top), so this
    # is safe to import in the parent. trajectory_generator is NOT imported
    # here — we use the inlined _load_trajectories above.
    from src.camera_setup import parse_cameras_yaml

    cfg = _load_yaml(args.config)
    camera_cfg = _load_yaml(args.cameras)
    cam_specs = parse_cameras_yaml(camera_cfg)
    camera_design = camera_cfg.get("camera_design", {"enable": False})
    workers = args.num_workers or cfg["runtime"].get("num_workers", 1)

    if camera_design.get("enable", False):
        print("Using ADAPTIVE camera planner (per-object best_yaw + jitter)")
    else:
        print("Using STATIC cameras from cameras.yaml")

    with open(args.object_list) as f:
        obj_by_id = {o["obj_id"]: o for o in json.load(f)["objects"]}

    trajs = _load_trajectories(args.trajectories)
    n_total = len(trajs)
    n_skipped_failed = 0
    n_skipped_no_obj = 0
    job_args = []
    for t in trajs:
        if args.only_success and not t.get("success", False):
            n_skipped_failed += 1
            continue
        obj = obj_by_id.get(t["obj_id"])
        if obj is None:
            n_skipped_no_obj += 1
            continue
        job_args.append((t, obj, cam_specs, camera_design, cfg, args.out))

    if args.limit is not None:
        job_args = job_args[: args.limit]

    print(f"Loaded {n_total} trajectories.")
    if args.only_success:
        print(f"  filtered out {n_skipped_failed} with success=False")
    if n_skipped_no_obj:
        print(f"  filtered out {n_skipped_no_obj} with missing object_list entry")
    print(f"Rendering {len(job_args)} trajectories with {workers} workers "
          f"across {args.n_gpu} GPUs (pid %% n_gpu pinning)...")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    n_done = n_ok = n_skip = n_fail = 0
    t_start = time.time()
    times: list[float] = []

    if workers <= 1:
        _init_worker(args.n_gpu)
        for ja in job_args:
            tid, status, dt = _worker(ja)
            n_done += 1
            times.append(dt)
            if status == "ok":
                n_ok += 1
            elif status == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  {tid}: {status}")
            if n_done % 10 == 0 or n_done == len(job_args):
                avg = sum(times[-50:]) / max(len(times[-50:]), 1)
                eta_s = avg * (len(job_args) - n_done)
                print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} "
                      f"fail={n_fail}  avg={avg:.1f}s  eta={eta_s/60:.1f}min")
    else:
        # Use fork (Linux default). The parent has NOT imported sapien
        # (we deliberately avoid top-level sapien imports above), so each
        # forked child can set CUDA_VISIBLE_DEVICES in _init_worker BEFORE
        # `_worker` lazy-imports src.renderer (and thus sapien).
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(args.n_gpu,),
        ) as ex:
            futures = {ex.submit(_worker, ja): ja[0]["traj_id"] for ja in job_args}
            for fut in as_completed(futures):
                tid = futures[fut]
                try:
                    tid, status, dt = fut.result()
                    times.append(dt)
                except Exception as e:  # noqa: BLE001
                    status = f"error: {type(e).__name__}: {e}"
                n_done += 1
                if status == "ok":
                    n_ok += 1
                elif status == "skipped":
                    n_skip += 1
                else:
                    n_fail += 1
                    print(f"  {tid}: {status}")
                if n_done % 25 == 0 or n_done == len(job_args):
                    avg = sum(times[-100:]) / max(len(times[-100:]), 1)
                    elapsed = time.time() - t_start
                    eta_s = (avg * (len(job_args) - n_done)) / max(workers, 1)
                    print(f"[{n_done}/{len(job_args)}] ok={n_ok} skip={n_skip} "
                          f"fail={n_fail}  avg={avg:.1f}s/traj  "
                          f"elapsed={elapsed/60:.1f}min  eta={eta_s/60:.1f}min")

    total = time.time() - t_start
    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail} "
          f"in {total/60:.1f}min  ({total/max(n_done,1):.1f}s/traj wall avg)")
    print(f"Output in {args.out}")


if __name__ == "__main__":
    main()
