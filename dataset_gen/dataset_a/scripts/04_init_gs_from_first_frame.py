"""Step 4: For every rendered trajectory, take the first frame from each
camera, run the configured GS backend, and save <traj_id>/init_gs.ply.

Reads the rendered output of Step 3 (data/<traj_id>/{front,side,...}.mp4 +
cameras.json). Writes init_gs.ply alongside.

Backends are configured in configs/default.yaml#gs.backend:
  mvsplat | gsplat | hybrid
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.init_gs_backend import make_backend, save_gs_to_ply
from src.object_selector import load_yaml


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def read_first_frame_mp4(path: Path) -> np.ndarray:
    """Return the first RGB frame as (H, W, 3) uint8."""
    import imageio
    reader = imageio.get_reader(str(path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def load_cameras_for_traj(traj_dir: Path) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """Returns (camera_names, list_of_K, list_of_c2w) in order."""
    with open(traj_dir / "cameras.json") as f:
        cams = json.load(f)
    names = sorted(cams.keys())  # deterministic order
    Ks, c2ws = [], []
    for name in names:
        K = _intrinsics_to_3x3(cams[name]["intrinsics"])
        w2c = np.array(cams[name]["extrinsics"]["world_to_camera_4x4"], dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        Ks.append(K.astype(np.float32))
        c2ws.append(c2w.astype(np.float32))
    return names, Ks, c2ws


def _intrinsics_to_3x3(intr: dict) -> np.ndarray:
    fx, fy = intr["fx"], intr["fy"]
    cx, cy = intr["cx"], intr["cy"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


# ----------------------------------------------------------------------------
# per-trajectory worker
# ----------------------------------------------------------------------------
def process_one(args_tuple):
    traj_dir, gs_cfg, device, force = args_tuple
    # Round-robin GPU assignment: worker process N uses GPU (N % n_gpu)
    import os
    if device.startswith("cuda") and ":" not in device:
        n_gpu = int(os.environ.get("WORLD_GPU_COUNT", "1"))
        if n_gpu > 1:
            gpu_id = os.getpid() % n_gpu
            device = f"cuda:{gpu_id}"
            # Note: don't set CUDA_VISIBLE_DEVICES here — torch is already
            # imported in this worker, so it has no effect, and combined with
            # device="cuda:N" causes "invalid device ordinal" errors.
    traj_dir = Path(traj_dir)
    out_path = traj_dir / "init_gs.ply"
    if out_path.exists() and not force:
        return traj_dir.name, "skipped"

    try:
        names, Ks, c2ws = load_cameras_for_traj(traj_dir)
        rgbs = [read_first_frame_mp4(traj_dir / f"{n}.mp4") for n in names]
    except Exception as e:  # noqa: BLE001
        return traj_dir.name, f"read-error: {e}"

    # If backend is "mesh", bypass gsplat — sample directly from PartNet mesh
    if gs_cfg.get("backend") == "mesh":
        from src.init_gs_mesh import reconstruct_from_mesh
        meta_path = traj_dir / "meta.json"
        try:
            meta = json.loads(meta_path.read_text())

            # Soft-body trajectories use procedural meshes (no PartNet folder).
            # Sample from the soft_object_spec stored in meta.
            if meta.get("object_type") == "soft" or not meta.get("obj_folder"):
                soft_spec = meta.get("soft_object_spec")
                if soft_spec:
                    from src.soft_objects import make_from_spec, SoftObjectSpec
                    from src.init_gs_mesh import _sample_to_gs_dict
                    rest_mesh = make_from_spec(SoftObjectSpec.from_dict(soft_spec))
                    gs = _sample_to_gs_dict(rest_mesh, gs_cfg.get("fallback_init_n_points", 50000))
                    save_gs_to_ply(gs, out_path)
                    return traj_dir.name, "ok"
                # No spec either — skip
                return traj_dir.name, f"backend-error: no PartNet folder and no soft_object_spec"

            obj_folder = meta.get("obj_folder")
            if not obj_folder:
                obj_id = meta.get("obj_id")
                import os
                partnet_root = os.environ.get("PARTNET_MOBILITY_ROOT", "")
                if partnet_root:
                    obj_folder = os.path.join(partnet_root, "dataset", obj_id)
            if not obj_folder or not Path(obj_folder).exists():
                return traj_dir.name, f"backend-error: cannot find obj_folder for {meta.get('obj_id')}"

            # Read first-frame joint qpos so the mesh is sampled at the
            # actual starting state (e.g. door already open for "close" task)
            first_qpos = None
            joint_name = meta.get("joint_name")
            traj_npz = traj_dir / "trajectory.npz"
            if traj_npz.exists():
                try:
                    data = np.load(traj_npz)
                    qpos_seq = data.get("joint_qpos")
                    if qpos_seq is not None and len(qpos_seq) > 0:
                        # joint_qpos in trajectory.npz is shape (n_frames,) for single-joint
                        first_qpos = float(qpos_seq[0])
                except Exception:
                    pass

            gs = reconstruct_from_mesh(
                obj_folder,
                n_points=gs_cfg.get("fallback_init_n_points", 50000),
                joint_qpos=first_qpos,
                joint_name=joint_name,
            )
        except Exception as e:
            return traj_dir.name, f"backend-error: {e}"
    else:
        backend = make_backend(gs_cfg, device=device)
        try:
            gs = backend.reconstruct(rgbs, Ks, c2ws)
        except Exception as e:  # noqa: BLE001
            return traj_dir.name, f"backend-error: {e}"
    if gs is None:
        return traj_dir.name, "backend returned None"

    save_gs_to_ply(gs, out_path)
    return traj_dir.name, "ok"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Output dir from Step 3 (renderer); each subdir is a traj")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--backend", default=None,
                    help="Override config gs.backend: mvsplat | gsplat | hybrid")
    ap.add_argument("--num_workers", type=int, default=1,
                    help="Parallel workers; each loads its own backend (memory cost)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if init_gs.ply already exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N trajectories (for smoke test)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = load_yaml(args.config)
    gs_cfg = dict(cfg["gs"])
    if args.backend:
        gs_cfg["backend"] = args.backend

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    traj_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if args.limit:
        traj_dirs = traj_dirs[: args.limit]
    print(f"Processing {len(traj_dirs)} trajectories with backend={gs_cfg['backend']}")

    job_args = [(str(d), gs_cfg, args.device, args.force) for d in traj_dirs]

    n_ok = n_skip = n_fail = 0
    if args.num_workers <= 1:
        for ja in job_args:
            tid, status = process_one(ja)
            print(f"  {tid}: {status}")
            if status == "ok":           n_ok += 1
            elif status == "skipped":    n_skip += 1
            else:                        n_fail += 1
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(process_one, ja): ja[0] for ja in job_args}
            for fut in as_completed(futures):
                tid_path = futures[fut]
                try:
                    tid, status = fut.result()
                except Exception as e:  # noqa: BLE001
                    tid, status = Path(tid_path).name, f"fatal: {e}"
                print(f"  {tid}: {status}")
                if status == "ok":           n_ok += 1
                elif status == "skipped":    n_skip += 1
                else:                        n_fail += 1

    print(f"\nDone. ok={n_ok}, skipped={n_skip}, failed={n_fail}")


if __name__ == "__main__":
    main()
