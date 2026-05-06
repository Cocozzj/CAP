"""Visual metrics that require rendering: PSNR, LPIPS, SSIM, multi-view consistency.

For each baseline that produced ``pred_4dgs.npz``:
  1. Load 4DGS sequence
  2. Render through gsplat at each camera in ``cameras.json``
  3. Compare rendered frames to GT video frames
  4. Update metrics.json with PSNR / LPIPS / SSIM

For MAGVIT v2 (no 4DGS, only ``pred_render.mp4``):
  - Read pred_render.mp4 directly
  - Compare to GT cam0.mp4
  - 3-view consistency: N/A (it generated only one view)

Usage:

    python -m eval.baseline.render_metrics \\
        --baselines tamp_pddl physgaussian physdreamer magvit_v2 motiongpt ours \\
        --output-root runs/baselines \\
        --data-root  dataset \\
        --T 30 --image-size 256

Then re-run aggregate.py to pull these metrics into the main table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from .common import GS4DSequence, TrajMetrics


# ══════════════════════════════════════════════════════════════════════
# Lazy LPIPS / gsplat imports
# ══════════════════════════════════════════════════════════════════════

_LPIPS_NET = None
def _get_lpips(device):
    """Lazily build the LPIPS evaluator on first use."""
    global _LPIPS_NET
    if _LPIPS_NET is None:
        import lpips
        _LPIPS_NET = lpips.LPIPS(net="alex").to(device).eval()
    return _LPIPS_NET


def _try_import_gsplat():
    try:
        import gsplat
        return gsplat
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════
# Image-level metrics
# ══════════════════════════════════════════════════════════════════════

def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Per-image PSNR.  Both arrays in [0, 1] float, [H, W, 3]."""
    mse = float(((pred - gt) ** 2).mean())
    if mse < 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM via skimage.  Both arrays in [0, 1] float, [H, W, 3]."""
    try:
        from skimage.metrics import structural_similarity as _ssim
    except ImportError:
        return float("nan")
    return float(_ssim(gt, pred, channel_axis=-1, data_range=1.0))


def lpips_score(pred: np.ndarray, gt: np.ndarray, device="cuda") -> float:
    """LPIPS distance (lower = more similar)."""
    net = _get_lpips(device)
    p = torch.from_numpy(pred).permute(2, 0, 1).float().unsqueeze(0).to(device)
    g = torch.from_numpy(gt).permute(2, 0, 1).float().unsqueeze(0).to(device)
    p = p * 2 - 1; g = g * 2 - 1                          # LPIPS expects [-1, 1]
    with torch.no_grad():
        return float(net(p, g).item())


def multi_view_consistency(rendered: np.ndarray) -> float:
    """L2 difference between views' rendered frames, averaged across time.

    rendered: [V, T, H, W, 3]   per-view rendered frames
    Higher = more inconsistent.  3DGS-based methods produce LOW values by
    construction (shared scene); pixel-level methods produce HIGH values.
    """
    if rendered.shape[0] < 2:
        return float("nan")
    V = rendered.shape[0]
    diffs = []
    for i in range(V):
        for j in range(i + 1, V):
            diffs.append(float(np.abs(rendered[i] - rendered[j]).mean()))
    return float(np.mean(diffs))


# ══════════════════════════════════════════════════════════════════════
# 4DGS → rendered video
# ══════════════════════════════════════════════════════════════════════

def render_4dgs_through_gsplat(
    seq:        GS4DSequence,
    intrinsics: np.ndarray,           # [V, 3, 3]
    extrinsics: np.ndarray,           # [V, 4, 4] world-to-camera
    image_size: int = 256,
    device:     str = "cuda",
) -> Optional[np.ndarray]:
    """Render a 4DGS sequence through every camera.  Returns [V, T, H, W, 3].

    Uses gsplat's rasterize_gaussians.  If gsplat isn't installed (or
    incompatible with the CUDA version on the lab server), returns None and
    the caller falls back to a placeholder.  See the wrapper below for
    how to install.
    """
    gsplat = _try_import_gsplat()
    if gsplat is None:
        return None

    V = int(intrinsics.shape[0])
    T = int(seq.T)
    out = np.zeros((V, T, image_size, image_size, 3), dtype=np.float32)

    mu      = torch.from_numpy(seq.mu).float().to(device)             # [T, N, 3]
    cov     = torch.from_numpy(seq.cov).float().to(device)            # [T, N, 3, 3]
    sh      = torch.from_numpy(seq.sh).float().to(device)             # [T, N, 48]
    opacity = torch.from_numpy(seq.opacity).float().to(device)        # [T, N, 1]

    K = torch.from_numpy(intrinsics).float().to(device)               # [V, 3, 3]
    w2c = torch.from_numpy(extrinsics).float().to(device)             # [V, 4, 4]

    # gsplat's exact API differs by version; this is a pseudo-rendering
    # placeholder that matches our model's renderer.py call signature.
    # On the lab server, replace this with the actual gsplat call you use
    # in eval/render_hook.py / model/executor/renderer.py.
    try:
        from model.executor.renderer import render_gs_sequence
    except ImportError:
        return None

    for v in range(V):
        try:
            frames = render_gs_sequence(
                mu=mu, cov=cov, sh=sh, opacity=opacity,
                K=K[v:v+1], w2c=w2c[v:v+1],
                image_size=image_size,
            )                                                         # [T, H, W, 3]
            out[v] = frames.detach().cpu().numpy()
        except Exception:
            return None
    return out


def load_gt_video_frames(traj_dir: Path, T: int, image_size: int = 256) -> Optional[np.ndarray]:
    """Read cam0/1/2.mp4, sub-sample to T frames each.  Returns [V, T, H, W, 3] in [0,1]."""
    try:
        import cv2
    except ImportError:
        return None
    cams = ["cam0", "cam1", "cam2"]
    out = []
    for cam in cams:
        p = traj_dir / f"{cam}.mp4"
        if not p.exists():
            continue
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n < 1:
            cap.release()
            continue
        idx = np.linspace(0, n - 1, T, dtype=int)
        frames = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, img = cap.read()
            if not ok:
                continue
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img.astype(np.float32) / 255.0)
        cap.release()
        if frames:
            out.append(np.stack(frames, axis=0))
    if not out:
        return None
    return np.stack(out, axis=0)                                       # [V, T, H, W, 3]


# ══════════════════════════════════════════════════════════════════════
# Main: iterate all (baseline, dataset, split, traj) tuples
# ══════════════════════════════════════════════════════════════════════

def evaluate_one(
    pred_npz_path: Path,
    traj_dir:      Path,
    image_size:    int = 256,
    T:             int = 30,
    device:        str = "cuda",
    is_pixel_only: bool = False,           # True for MAGVIT v2 path
) -> dict:
    """Compute visual metrics for one trajectory's prediction.

    Returns a dict with PSNR / LPIPS / SSIM / multi_view_consistency
    or {} if anything failed.
    """
    out: dict = {}

    # GT video frames (always needed)
    gt_frames = load_gt_video_frames(traj_dir, T=T, image_size=image_size)
    if gt_frames is None:
        return {}

    if is_pixel_only:
        # MAGVIT v2 path: read pred_render.mp4 (cam0 only)
        pred_video_path = pred_npz_path.parent / "pred_render.mp4"
        if not pred_video_path.exists():
            return {}
        try:
            import cv2
        except ImportError:
            return {}
        cap = cv2.VideoCapture(str(pred_video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = np.linspace(0, max(n - 1, 0), T, dtype=int)
        pred = []
        for fi in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, img = cap.read()
            if not ok:
                continue
            img = cv2.resize(img, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            pred.append(img)
        cap.release()
        if not pred:
            return {}
        pred = np.stack(pred, axis=0)                                # [T, H, W, 3]

        # Compare cam0 only
        gt_cam0 = gt_frames[0]                                        # [T, H, W, 3]
        psnrs   = [psnr(pred[t], gt_cam0[t]) for t in range(min(T, len(pred)))]
        lpipss  = [lpips_score(pred[t], gt_cam0[t], device) for t in range(min(T, len(pred)))]
        ssims   = [ssim(pred[t], gt_cam0[t]) for t in range(min(T, len(pred)))]
        out["psnr"]  = float(np.mean(psnrs))
        out["lpips"] = float(np.mean(lpipss))
        out["ssim"]  = float(np.mean(ssims))
        # Multi-view consistency: N/A for MAGVIT v2 (only generated cam0)
        return out

    # 4DGS path
    if not pred_npz_path.exists():
        return {}
    seq = GS4DSequence.load(pred_npz_path)

    # Read cameras.json for intrinsics/extrinsics
    cams_path = traj_dir / "cameras.json"
    if not cams_path.exists():
        return {}
    with open(cams_path) as f:
        cams = json.load(f)
    cam_names = [k for k in ("cam0", "cam1", "cam2") if k in cams]
    Ks   = []
    w2cs = []
    for name in cam_names:
        intr = cams[name]["intrinsics"]
        Ks.append(np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1],
        ], dtype=np.float32))
        w2cs.append(np.array(cams[name]["extrinsics"]["world_to_camera_4x4"],
                              dtype=np.float32))
    K = np.stack(Ks, axis=0)
    w2c = np.stack(w2cs, axis=0)

    rendered = render_4dgs_through_gsplat(
        seq, intrinsics=K, extrinsics=w2c,
        image_size=image_size, device=device,
    )
    if rendered is None:
        return {}

    # Per-frame PSNR / LPIPS / SSIM, averaged over T and V
    V_ = rendered.shape[0]
    T_ = min(rendered.shape[1], gt_frames.shape[1])
    psnrs, lpipss, ssims = [], [], []
    for v in range(V_):
        for t in range(T_):
            psnrs .append(psnr (rendered[v, t], gt_frames[v, t]))
            lpipss.append(lpips_score(rendered[v, t], gt_frames[v, t], device))
            ssims .append(ssim (rendered[v, t], gt_frames[v, t]))

    out["psnr"]  = float(np.mean(psnrs))
    out["lpips"] = float(np.mean(lpipss))
    out["ssim"]  = float(np.mean(ssims))
    # Cross-view consistency on the rendered frames themselves (lower = consistent)
    out["mv_consistency"] = multi_view_consistency(rendered[:, :T_])
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--baselines", nargs="+",
                   default=["tamp_pddl", "physgaussian", "svd",
                            "motiongpt", "ours"])
    p.add_argument("--output-root", default="runs/baselines")
    p.add_argument("--data-root", default="dataset")
    p.add_argument("--datasets", nargs="+", default=["dataset_a", "dataset_b"])
    p.add_argument("--T", type=int, default=30)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None,
                   help="for debugging; only do N trajectories per (baseline, dataset, split)")
    args = p.parse_args(argv)

    out_root  = Path(args.output_root)
    data_root = Path(args.data_root)
    n_total = n_ok = n_skip = 0
    t0 = time.time()

    for baseline in args.baselines:
        # Pixel-only baselines write pred_render.mp4 (no 3D output).  We
        # compare frames directly to GT cam0 instead of rendering preds.
        is_pixel_only = baseline in ("svd",)
        for dataset in args.datasets:
            base = out_root / baseline / dataset
            if not base.exists():
                continue
            for split_dir in sorted(base.iterdir()):
                if not split_dir.is_dir():
                    continue
                traj_dirs = sorted(split_dir.iterdir())
                if args.limit:
                    traj_dirs = traj_dirs[: args.limit]
                for traj_out_dir in traj_dirs:
                    if not traj_out_dir.is_dir():
                        continue
                    n_total += 1
                    gt_dir = data_root / dataset / "data" / traj_out_dir.name
                    if not gt_dir.exists():
                        n_skip += 1
                        continue
                    pred_npz = traj_out_dir / "pred_4dgs.npz"

                    visual = evaluate_one(
                        pred_npz, gt_dir,
                        image_size=args.image_size,
                        T=args.T, device=args.device,
                        is_pixel_only=is_pixel_only,
                    )
                    if not visual:
                        n_skip += 1
                        continue

                    # Merge into existing metrics.json
                    mp = traj_out_dir / "metrics.json"
                    m  = TrajMetrics.load(mp) if mp.exists() else TrajMetrics()
                    if "psnr"  in visual: m.psnr  = visual["psnr"]
                    if "lpips" in visual: m.lpips = visual["lpips"]
                    if "ssim"  in visual: m.ssim  = visual["ssim"]
                    m.save(mp)
                    n_ok += 1
                    if n_ok <= 3 or n_ok % 50 == 0:
                        print(f"  ✓ [{baseline}] {traj_out_dir.name}  "
                              f"psnr={visual.get('psnr', 0):.2f}  "
                              f"lpips={visual.get('lpips', 0):.4f}")

    print(f"\n=== render_metrics complete ===")
    print(f"  total:   {n_total}")
    print(f"  ok:      {n_ok}")
    print(f"  skipped: {n_skip}  (no GT video / render failed / no pred)")
    print(f"  elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
