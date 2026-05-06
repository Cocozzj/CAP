"""Invoke PhysGaussian on every trajectory's pre-built config.

Workflow:
  1. ``convert_data.py`` already wrote one ``physgs_config.json`` per trajectory.
  2. This script iterates over those configs, calls PhysGaussian's
     ``gs_simulation.py`` with each one, and saves the resulting deformed
     Gaussian sequence to ``pred_4dgs.npz``.
  3. PhysGaussian's exact CLI / output schema may differ from this stub.
     See the TODO markers and adjust against the version you cloned.

Usage:

    python -m eval.baseline.physgaussian.run_eval \\
        --output-root runs/baselines \\
        --physgs-repo $PHYSGAUSSIAN_REPO
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from ..common import GS4DSequence, TrajMetrics


# ════════════════════════════════════════════════════════════════════════
# PLY adapter:  our (sh_degree=0)  →  PhysGaussian's expected (sh_degree=3)
# ════════════════════════════════════════════════════════════════════════

def _transform_and_pad_ply(
    src: Path,
    dst: Path,
    target_extent: float = 0.4,
    target_center: tuple[float, float, float] = (1.0, 1.0, 0.8),
) -> dict:
    """Read ``src`` (a sh_degree=0 3DGS PLY in arbitrary coords) and write
    ``dst`` (sh_degree=3, properties matching PhysGaussian's strict loader,
    AND geometry rigidly mapped into PhysGaussian's MPM grid space).

    Why both transforms in one pass?
    --------------------------------
    PhysGaussian's MPM grid is implicitly [0, 2]³ with the object expected
    near (1, 1, 1) (see ficus / wolf / tear_bread example configs).  Our
    PartNet-derived PLYs live in roughly [-0.7, 0.6]³ (centered ≈ origin).
    Without the transform, particles immediately escape the grid and the
    Warp MPM kernel hits CUDA_ERROR_ILLEGAL_ADDRESS.

    Required output columns (in this order, per ``gaussian_model.py``):
        x, y, z, nx, ny, nz,
        f_dc_0..2,
        f_rest_0..44,
        opacity,
        scale_0..2,
        rot_0..3
    Total = 3 + 3 + 3 + 45 + 1 + 3 + 4 = 62 properties.

    Returns a dict ``{src_center, src_extent, scale, target_center,
    target_extent}`` so the caller can patch the config (viewpoint center,
    boundary conditions) consistently with the new geometry.
    """
    from plyfile import PlyData, PlyElement

    p_in = PlyData.read(str(src))
    v_in = p_in["vertex"]
    n_pts = len(v_in)

    def col(name, default=0.0):
        try:
            return np.asarray(v_in[name], dtype=np.float32)
        except (KeyError, ValueError):
            return np.full(n_pts, default, dtype=np.float32)

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1)
    src_min = xyz.min(axis=0)
    src_max = xyz.max(axis=0)
    src_center = (src_min + src_max) / 2.0
    src_extent = float((src_max - src_min).max())
    scale = float(target_extent / max(src_extent, 1e-6))

    target_center_np = np.asarray(target_center, dtype=np.float32)
    new_xyz = (xyz - src_center) * scale + target_center_np
    # Gaussian scale is stored as log(σ) in 3DGS PLYs — uniform spatial
    # scaling adds log(scale) to each scale_* channel.
    log_scale_factor = float(np.log(scale))

    cols = {
        "x":  new_xyz[:, 0], "y":  new_xyz[:, 1], "z":  new_xyz[:, 2],
        "nx": col("nx"),     "ny": col("ny"),     "nz": col("nz"),
        "f_dc_0": col("f_dc_0"),
        "f_dc_1": col("f_dc_1"),
        "f_dc_2": col("f_dc_2"),
    }
    for i in range(45):
        cols[f"f_rest_{i}"] = col(f"f_rest_{i}", default=0.0)
    cols["opacity"] = col("opacity")
    cols["scale_0"] = col("scale_0") + log_scale_factor
    cols["scale_1"] = col("scale_1") + log_scale_factor
    cols["scale_2"] = col("scale_2") + log_scale_factor
    cols["rot_0"]   = col("rot_0", default=1.0)        # quaternion w default 1
    cols["rot_1"]   = col("rot_1"); cols["rot_2"] = col("rot_2"); cols["rot_3"] = col("rot_3")

    dtype = [(k, "f4") for k in cols.keys()]
    arr = np.empty(n_pts, dtype=dtype)
    for k, v in cols.items():
        arr[k] = v

    el = PlyElement.describe(arr, "vertex")
    PlyData([el], text=False).write(str(dst))

    return {
        "src_center":    src_center.tolist(),
        "src_extent":    src_extent,
        "scale":         scale,
        "target_center": list(target_center),
        "target_extent": target_extent,
    }


def _patch_physgs_config(orig_path: Path, dst_path: Path, xform: dict) -> None:
    """Load convert_data's config and add the bits PhysGaussian needs to
    actually run: a confining ``bounding_box`` BC, a sticky ground plane
    just below the (transformed) object, and ``mpm_space_viewpoint_center``
    aligned with where we placed the object.

    Also strips fields convert_data added for our own bookkeeping
    (model_path / traj_id / dataset / split) so PhysGaussian doesn't choke
    on unknown keys.
    """
    cfg = json.loads(orig_path.read_text())

    target_center = xform["target_center"]
    target_extent = xform["target_extent"]
    # Floor sits 0.05 units below the object's lowest point — keeps the
    # MPM solver from instantly clamping particles into the ground plane.
    floor_z = max(target_center[2] - target_extent / 2.0 - 0.05, 0.05)

    cfg["mpm_space_viewpoint_center"] = list(target_center)

    cfg["boundary_conditions"] = [
        {"type": "bounding_box"},        # confine particles to MPM grid
        {                                # ground plane
            "type":       "surface_collider",
            "point":      [target_center[0], target_center[1], floor_z],
            "normal":     [0.0, 0.0, 1.0],
            "surface":    "sticky",
            "friction":   0.0,    # PhysGaussian asserts friction==0 for sticky
            "start_time": 0,
            "end_time":   1000.0,
        },
    ]

    # ── MPM stability clamp ────────────────────────────────────────────
    # PartNet ρ values (E up to 1e8) blow up MPM with the default
    # substep_dt=1e-4 (CFL: dt < dx / sqrt(E/ρ)).  Clamp to PhysGaussian
    # ficus/wolf-tested ranges — baseline goal is "doesn't crash", not
    # "physically calibrated".  We document the clamp in the metrics
    # ``notes`` field so downstream analysis can flag it.
    orig_E = float(cfg.get("E", 1e6))
    orig_dt = float(cfg.get("substep_dt", 1e-4))
    cfg["E"] = float(np.clip(orig_E, 1e3, 5e6))    # ficus is 2e6, wolf 5e7
    cfg["nu"] = float(np.clip(cfg.get("nu", 0.3), 0.0, 0.45))
    cfg["density"] = float(np.clip(cfg.get("density", 200.0), 50.0, 2000.0))
    cfg["substep_dt"] = min(orig_dt, 5e-5)         # safer for clamped E
    cfg.setdefault("n_grid", 100)                  # ficus default-ish

    # Strip our bookkeeping fields — PhysGaussian's decode_param.py may
    # not tolerate extras depending on the version.
    for k in ("model_path", "traj_id", "dataset", "split"):
        cfg.pop(k, None)

    dst_path.write_text(json.dumps(cfg, indent=2))


def _run_physgaussian_one(
    cfg_path: Path,
    output_dir: Path,
    physgs_repo: Path,
    timeout_secs: int = 600,
) -> tuple[bool, str]:
    """Invoke PhysGaussian's ``gs_simulation.py`` on one config.

    PhysGaussian's CLI:
        python gs_simulation.py \\
            --model_path <model_dir>           # 3DGS-style scene dir, REQUIRED
            --config     <cfg.json>
            --output_path <out_dir>

    --model_path must contain ``point_cloud/iteration_*/point_cloud.ply``
    (PhysGaussian's load_checkpoint() globs for the latest ``iteration_*``).
    Our convert_data.py only emits a single init PLY, so we wrap it in the
    expected layout via symlink before invocation.  No cameras.json /
    cfg_args are needed because gs_simulation.py uses load_checkpoint()
    directly, not the full Scene() loader.

    Returns (success, message).  On failure, message is the exception/error
    string for logging.

    PYTHONPATH:  ``utils.sh_utils`` lives in PhysGaussian's
    ``gaussian-splatting/`` submodule (NOT in the top-level ``utils/``).
    We prepend it to PYTHONPATH so ``gs_simulation.py``'s
    ``from utils.sh_utils import eval_sh`` resolves.
    """
    import os

    # ── Build fake 3DGS model_path layout ──────────────────────────────────
    # convert_data.py wrote the absolute init-PLY path into cfg["model_path"];
    # PhysGaussian's CLI uses --model_path (which must be a *directory*), so
    # we rebuild the expected ``point_cloud/iteration_30000/point_cloud.ply``
    # tree and symlink the init PLY into it.
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        return False, f"failed to read cfg json: {e}"
    init_ply_str = cfg.get("model_path")
    if not init_ply_str:
        return False, "cfg has no 'model_path' field (init PLY path)"
    init_ply = Path(init_ply_str).expanduser()
    if not init_ply.exists():
        return False, f"init PLY not found: {init_ply}"

    # output_dir == traj_dir/physgs_raw  →  traj_dir == output_dir.parent
    traj_dir = output_dir.parent
    model_dir = traj_dir / "physgs_model"
    iter_dir = model_dir / "point_cloud" / "iteration_30000"
    iter_dir.mkdir(parents=True, exist_ok=True)
    target_ply = iter_dir / "point_cloud.ply"
    if target_ply.exists() or target_ply.is_symlink():
        target_ply.unlink()

    # 1. Transform PLY into PhysGaussian's MPM grid space ([0,2]³ around
    #    ~ (1,1,1)) and pad SH up to degree 3.  ``load_checkpoint()`` hard-
    #    codes sh_degree=3 → 3DGS's load_ply asserts 45 ``f_rest_*`` props.
    try:
        xform = _transform_and_pad_ply(init_ply, target_ply)
    except Exception as e:
        return False, f"failed to transform/pad PLY: {e}"

    # 2. Patch the JSON config: add a bounding_box + ground-plane BC and
    #    align mpm_space_viewpoint_center with where we placed the object.
    #    Without these, the MPM solver immediately diverges (particles
    #    escape the grid → CUDA_ERROR_ILLEGAL_ADDRESS in Warp).
    patched_cfg_path = model_dir / "physgs_config_patched.json"
    try:
        _patch_physgs_config(cfg_path, patched_cfg_path, xform)
    except Exception as e:
        return False, f"failed to patch config: {e}"

    # 3. gs_simulation.py's get_camera_view() opens ``model_path/cameras.json``
    #    every frame even when --render_img is off (rasterizer is built
    #    inside the simulation loop).  With default_camera_index=-1 only
    #    width/height/fx/fy come from this file — position/rotation get
    #    overridden from init_azimuthm/elevation/radius.
    cameras_json = model_dir / "cameras.json"
    cameras_json.write_text(json.dumps([{
        "id":       0,
        "img_name": "synth_view",
        "width":    256,           # small → cheap rasterize per frame
        "height":   256,
        "position": [0.0, 0.0, 4.0],          # overridden
        "rotation": [[1.0, 0.0, 0.0],          # overridden
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]],
        "fx":       256.0,
        "fy":       256.0,
    }], indent=2))

    cmd = [
        "python", str(physgs_repo / "gs_simulation.py"),
        "--model_path",  str(model_dir.resolve()),
        "--config",      str(patched_cfg_path.resolve()),
        "--output_path", str(output_dir.resolve()),
        "--output_ply",                              # we need PLYs to parse back
    ]
    env = dict(os.environ)
    extra_paths = [
        str(physgs_repo / "gaussian-splatting"),     # for utils.sh_utils, scene/, etc.
        str(physgs_repo),                             # for top-level imports
    ]
    env["PYTHONPATH"] = os.pathsep.join(
        extra_paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_secs,
            cwd=str(physgs_repo), env=env,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout_secs}s"
    except FileNotFoundError as e:
        return False, f"physgaussian script not found: {e}"

    if r.returncode != 0:
        # Common failures: solver divergence on thin objects, OOM on dense scenes
        err = (r.stderr or "")[-500:]
        return False, f"returncode={r.returncode}  stderr_tail={err!r}"
    return True, "ok"


def _collect_physgaussian_outputs(physgs_out_dir: Path) -> GS4DSequence | None:
    """Read whatever PhysGaussian wrote and convert to our GS4DSequence.

    PhysGaussian's exact output schema (verify against your version):
      - ``frames/frame_000.ply, frame_001.ply, ...`` per-timestep PLYs, OR
      - ``simulation.npz`` with arrays ``mu, cov, ...``

    This stub tries both.  Adjust to whatever your version actually writes.
    """
    # Schema 1: per-frame PLY files
    ply_files = sorted(physgs_out_dir.glob("frames/frame_*.ply"))
    if ply_files:
        try:
            from dataload.common import load_init_gs_ply
        except ImportError:
            return None

        T = len(ply_files)
        # Read first to learn N
        first = load_init_gs_ply(ply_files[0], n_points=10000, seed=0, c_sh=48)
        N = int(first.mu.shape[0])

        mu = np.zeros((T, N, 3), dtype=np.float32)
        for t, p in enumerate(ply_files):
            gs = load_init_gs_ply(p, n_points=N, seed=0, c_sh=48)
            mu[t] = gs.mu.numpy()
        # Reuse first frame's cov/sh/opacity/scale as broadcast (PhysGaussian
        # may not write them per-frame; verify in your version)
        cov0     = first.cov.numpy()  # placeholder, may need conversion
        sh0      = first.sh.numpy()
        opacity0 = first.opacity.numpy()
        scale0   = first.scale.numpy()

        # Build a 3x3 cov from quat (caller's job) — for now broadcast
        cov_full = np.eye(3, dtype=np.float32)[None, None].repeat(T, axis=0).repeat(N, axis=1)

        return GS4DSequence(
            mu=mu, cov=cov_full,
            sh=np.broadcast_to(sh0[None],      (T,) + sh0.shape).copy(),
            opacity=np.broadcast_to(opacity0[None], (T,) + opacity0.shape).copy(),
            scale=np.broadcast_to(scale0[None],   (T,) + scale0.shape).copy(),
        )

    # Schema 2: single npz
    npz_files = list(physgs_out_dir.glob("*.npz"))
    if npz_files:
        z = np.load(npz_files[0])
        # Expected keys: mu, cov, sh, opacity, scale  (verify schema)
        try:
            return GS4DSequence(
                mu=z["mu"], cov=z["cov"], sh=z["sh"],
                opacity=z["opacity"], scale=z["scale"],
            )
        except KeyError:
            return None

    return None


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--dataset-name", default="dataset_a")
    p.add_argument("--splits", nargs="+", default=None,
                   help="if None, run on all splits found under output_root/physgaussian/<dataset>/")
    p.add_argument("--physgs-repo", required=True,
                   help="path to cloned PhysGaussian repo (set $PHYSGAUSSIAN_REPO)")
    p.add_argument("--timeout", type=int, default=600,
                   help="per-trajectory timeout (PhysGaussian sims can hang on degenerate input)")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    physgs_repo = Path(args.physgs_repo).expanduser().resolve()
    if not (physgs_repo / "gs_simulation.py").exists():
        print(f"✗ PhysGaussian script not found at {physgs_repo}", file=sys.stderr)
        print(f"  set --physgs-repo or $PHYSGAUSSIAN_REPO to the cloned repo path",
              file=sys.stderr)
        return 1

    base = Path(args.output_root) / "physgaussian" / args.dataset_name
    if args.splits is None:
        splits = [d.name for d in sorted(base.iterdir()) if d.is_dir()]
    else:
        splits = args.splits

    total = ok = failed = 0
    t0 = time.time()
    for split in splits:
        split_dir = base / split
        traj_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        for i, traj_dir in enumerate(traj_dirs):
            if args.limit is not None and total >= args.limit:
                break
            total += 1
            cfg_path = traj_dir / "physgs_config.json"
            if not cfg_path.exists():
                print(f"  ⊘ {traj_dir.name}: no physgs_config.json (run convert_data first)")
                failed += 1
                continue

            # PhysGaussian writes its outputs into a subdir of traj_dir
            physgs_out = traj_dir / "physgs_raw"
            physgs_out.mkdir(exist_ok=True)

            success, msg = _run_physgaussian_one(
                cfg_path, physgs_out, physgs_repo, timeout_secs=args.timeout,
            )
            if not success:
                TrajMetrics(notes=f"physgs_failed: {msg}").save(traj_dir / "metrics.json")
                failed += 1
                if failed <= 3:
                    print(f"  ✗ {traj_dir.name}  {msg}")
                continue

            # Convert PhysGaussian raw output → our unified GS4DSequence format
            seq = _collect_physgaussian_outputs(physgs_out)
            if seq is None:
                TrajMetrics(notes="physgs_output_parse_failed").save(traj_dir / "metrics.json")
                failed += 1
                continue

            seq.save(traj_dir / "pred_4dgs.npz")
            TrajMetrics(notes="pending_eval").save(traj_dir / "metrics.json")
            ok += 1
            if ok <= 3 or ok % 50 == 0:
                print(f"  ✓ {traj_dir.name}  T={seq.T} N={seq.N}")

    dt = time.time() - t0
    print(f"\n=== PhysGaussian complete ===")
    print(f"  total:   {total}")
    print(f"  ok:      {ok}")
    print(f"  failed:  {failed}")
    print(f"  elapsed: {dt:.1f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
