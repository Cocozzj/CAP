"""PyTorch Dataset / DataLoader for Dataset-B (real-world single-view video).

Returns the SAME dict shape as Dataset-A's dataloader so the same training
code can consume both. Differences from Dataset-A:
    * V = 1 (only cam0)            -> frames shape [1, T, 3, H, W]
    * Optional `gt_depth` field     -> [1, 1, H, W] from DepthAnything (only t=0)
    * extrinsics = identity         -> single-view; no real world->cam transform
    * Verb phrase uses SSv2 placeholders so the noun is the actual object
      (e.g. "open the bottle" instead of "open the ssv2_realworld")

Usage:
    from src.dataloader import DatasetB, collate_fn
    ds = DatasetB(manifest_path='outputs/manifest.json',
                  data_dir='outputs/data',
                  split='train',
                  T=30, n_gs_points=10000,
                  return_gt_depth=True)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================
# GSParameter (identical to Dataset-A's)
# ============================================================================
@dataclass
class GSParameter:
    mu: torch.Tensor
    cov: torch.Tensor
    scale: torch.Tensor
    sh: torch.Tensor
    opacity: torch.Tensor

    def __len__(self) -> int:
        return int(self.mu.shape[0])

    def to(self, device, non_blocking: bool = False) -> "GSParameter":
        return GSParameter(
            mu=self.mu.to(device, non_blocking=non_blocking),
            cov=self.cov.to(device, non_blocking=non_blocking),
            scale=self.scale.to(device, non_blocking=non_blocking),
            sh=self.sh.to(device, non_blocking=non_blocking),
            opacity=self.opacity.to(device, non_blocking=non_blocking),
        )


# ============================================================================
# Verb-phrase generation
# ============================================================================
_VERB_TEMPLATES = {
    "open":    "open the {obj}",
    "close":   "close the {obj}",
    "pull":    "pull the {obj}",
    "push":    "push the {obj}",
    "rotate":  "rotate the {obj}",
    "squeeze": "squeeze the {obj}",
    "fold":    "fold the {obj}",
    "pour":    "pour from the {obj}",
}

_DEFAULT_OBJECT = "object"


def task_to_text(task_name: str, raw_label: str = "", placeholders: Optional[List[str]] = None) -> str:
    """Build a natural-language verb phrase for a Dataset-B clip.

    Priority:
      1. If `placeholders` is non-empty, use placeholders[0] (= the actual
         object name annotated in SSv2, e.g. 'bottle', 'sock').
      2. Else, fall back to a generic 'object'.
    """
    obj = _DEFAULT_OBJECT
    if placeholders:
        # SSv2 placeholders are the actual nouns, lowercase and short
        obj = str(placeholders[0]).strip().lower() or _DEFAULT_OBJECT
    tpl = _VERB_TEMPLATES.get(task_name, task_name + " the {obj}")
    return tpl.format(obj=obj)


# ============================================================================
# Video loading (decord preferred, imageio fallback)
# ============================================================================
def _load_video_frames(mp4_path: Path,
                       frame_indices: Sequence[int],
                       target_size: Optional[int] = None) -> torch.Tensor:
    """Load specific frames from an mp4. Returns Tensor (T, 3, H, W) float32 in [0,1]."""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(str(mp4_path))
        arr = vr.get_batch(list(frame_indices)).asnumpy()
    except ImportError:
        import imageio
        reader = imageio.get_reader(str(mp4_path))
        try:
            arr = np.stack([reader.get_data(int(i)) for i in frame_indices], axis=0)
        finally:
            reader.close()

    if target_size is not None and arr.shape[1] != target_size:
        from PIL import Image
        out = np.empty((arr.shape[0], target_size, target_size, 3), dtype=np.uint8)
        for t in range(arr.shape[0]):
            out[t] = np.array(
                Image.fromarray(arr[t]).resize((target_size, target_size), Image.BILINEAR)
            )
        arr = out

    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (0, 3, 1, 2))
    return torch.from_numpy(arr)


# ============================================================================
# PLY loading (re-uses Dataset-A's logic — same standard-3DGS field layout)
# ============================================================================
_SH_C0 = 0.28209479177387814


def load_init_gs_ply(ply_path: Path,
                     n_points: Optional[int] = None,
                     seed: int = 0,
                     sh_degree: int = 3) -> GSParameter:
    """Read a Dataset-B init_gs.ply (or any standard 3DGS PLY) into GSParameter."""
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise RuntimeError("plyfile required. pip install plyfile") from e

    ply = PlyData.read(str(ply_path))
    v = ply["vertex"].data
    n = len(v)
    fields = set(v.dtype.names)

    mu = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if "opacity" in fields:
        op = np.asarray(v["opacity"], dtype=np.float32)
    else:
        op = np.ones(n, dtype=np.float32)
    if op.min() < -0.05 or op.max() > 1.05:
        op = 1.0 / (1.0 + np.exp(-op))
    opacity = np.clip(op, 0, 1)[:, None]

    if all(f"scale_{i}" in fields for i in range(3)):
        scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1).astype(np.float32)
    else:
        scale = np.full((n, 3), np.log(0.025), dtype=np.float32)

    if all(f"rot_{i}" in fields for i in range(4)):
        cov = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=-1).astype(np.float32)
    else:
        cov = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    qn = np.linalg.norm(cov, axis=-1, keepdims=True) + 1e-8
    cov = cov / qn

    c_sh = (sh_degree + 1) ** 2 * 3
    sh = np.zeros((n, c_sh), dtype=np.float32)
    if all(f"f_dc_{i}" in fields for i in range(3)):
        for i in range(3):
            sh[:, i] = np.asarray(v[f"f_dc_{i}"], dtype=np.float32)
        for i in range(c_sh - 3):
            key = f"f_rest_{i}"
            if key in fields:
                sh[:, 3 + i] = np.asarray(v[key], dtype=np.float32)

    if n_points is not None and n > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_points, replace=False)
        idx.sort()
        mu, cov, scale, sh, opacity = mu[idx], cov[idx], scale[idx], sh[idx], opacity[idx]

    return GSParameter(
        mu=torch.from_numpy(np.ascontiguousarray(mu)),
        cov=torch.from_numpy(np.ascontiguousarray(cov)),
        scale=torch.from_numpy(np.ascontiguousarray(scale)),
        sh=torch.from_numpy(np.ascontiguousarray(sh)),
        opacity=torch.from_numpy(np.ascontiguousarray(opacity)),
    )


# ============================================================================
# Camera + depth loading
# ============================================================================
def load_cameras(cameras_json_path: Path,
                 camera_names: Sequence[str]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    with open(cameras_json_path) as f:
        cams = json.load(f)
    Ks, w2cs = [], []
    for name in camera_names:
        c = cams[name]
        intr = c["intrinsics"]
        K = np.array([
            [intr["fx"], 0, intr["cx"]],
            [0, intr["fy"], intr["cy"]],
            [0, 0, 1],
        ], dtype=np.float32)
        w2c = np.array(c["extrinsics"]["world_to_camera_4x4"], dtype=np.float32)
        Ks.append(K)
        w2cs.append(w2c)
    return (torch.from_numpy(np.stack(Ks)), torch.from_numpy(np.stack(w2cs)))


def load_depth_first_frame(depth_npz_path: Path,
                           image_size: int) -> torch.Tensor:
    """Load the first-frame depth, return Tensor (1, H, W) float32."""
    d = np.load(depth_npz_path)
    depth = d["depth"].astype(np.float32)
    if depth.shape[0] != image_size or depth.shape[1] != image_size:
        from PIL import Image
        depth = np.asarray(
            Image.fromarray(depth).resize((image_size, image_size), Image.BILINEAR)
        )
    return torch.from_numpy(depth[None, :, :])  # (1, H, W)


# ============================================================================
# Dataset
# ============================================================================
class DatasetB(Dataset):
    """Dataset-B loader (real-world single-view video).

    Args:
        manifest_path:   outputs/manifest.json from Step 5
        data_dir:        outputs/data
        split:           'train' | 'val' | 'test'
        T:               frames per clip (default 30; constraint T>=10, T%5==0)
        n_gs_points:     subsample init_gs.ply to this many points
        image_size:      resize frames to this square (clips already 256;
                         no-op unless you want a different model input size)
        return_gt_frames: include `gt_frames` (= frames; reconstruction loss)
        return_cameras:   include `intrinsics` / `extrinsics` (single view)
        return_gt_depth:  include `gt_depth` of shape [1, 1, H, W] from
                         DepthAnything-v2 first-frame estimate
        return_task_id:   include integer task id
    """

    V_CAMERAS = ("cam0",)   # Dataset-B is single-view by construction

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir: Union[str, Path],
        split: str = "train",
        T: int = 30,
        n_gs_points: int = 10000,
        image_size: int = 256,
        return_gt_frames: bool = False,
        return_cameras: bool = False,
        return_gt_depth: bool = False,
        return_task_id: bool = False,
        require_text: bool = True,
        sh_degree: int = 3,
        seed: int = 0,
    ):
        if T < 10 or T % 5 != 0:
            raise ValueError(f"T must be >= 10 and a multiple of 5; got T={T}")
        if n_gs_points < 100:
            raise ValueError(f"n_gs_points={n_gs_points} too few; recommend >= 1000")

        self.data_dir = Path(data_dir)
        self.T = T
        self.n_gs_points = n_gs_points
        self.image_size = image_size
        self.return_gt_frames = return_gt_frames
        self.return_cameras = return_cameras
        self.return_gt_depth = return_gt_depth
        self.return_task_id = return_task_id
        self.require_text = require_text
        self.sh_degree = sh_degree
        self.seed = seed
        self.split = split

        with open(manifest_path) as f:
            manifest = json.load(f)
        self.entries = [e for e in manifest["entries"]
                        if split in e.get("splits", []) or e.get("split") == split]
        if not self.entries:
            raise ValueError(
                f"No entries for split={split!r}. Available splits: "
                f"{sorted({s for e in manifest['entries'] for s in e.get('splits', [])})}"
            )

        if return_task_id:
            unique_tasks = sorted(set(e["task_name"] for e in self.entries))
            self.task_to_id = {t: i for i, t in enumerate(unique_tasks)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        entry = self.entries[i]
        traj_dir = self.data_dir / entry["rel_dir"]
        n_total = int(entry.get("n_frames", 30))

        # Uniform frame sampling
        frame_indices = np.linspace(0, max(n_total - 1, 0), self.T, dtype=int).tolist()

        # Single camera; output shape (1, T, 3, H, W) to match Dataset-A's V dim
        v_frames = _load_video_frames(traj_dir / "cam0.mp4", frame_indices,
                                       target_size=self.image_size)
        frames = v_frames.unsqueeze(0).clamp(0, 1)

        gs = load_init_gs_ply(
            traj_dir / "init_gs.ply",
            n_points=self.n_gs_points,
            seed=self.seed + i,
            sh_degree=self.sh_degree,
        )

        # Verb phrase using SSv2 placeholders if present in meta
        meta_path = traj_dir / "meta.json"
        if self.require_text and meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            text = task_to_text(
                entry["task_name"],
                raw_label=meta.get("raw_label", ""),
                placeholders=meta.get("placeholders", []),
            )
        else:
            text = ""

        out: Dict = {
            "frames":    frames,
            "gs_params": gs,
            "text":      text,
        }

        if self.return_gt_frames:
            out["gt_frames"] = frames

        if self.return_cameras:
            K, w2c = load_cameras(traj_dir / "cameras.json", self.V_CAMERAS)
            out["intrinsics"] = K
            out["extrinsics"] = w2c

        if self.return_gt_depth:
            d_path = traj_dir / "depth.npz"
            if d_path.exists():
                # (1, H, W) -> (V=1, T=1, 1, H, W) so it stacks like frames
                d = load_depth_first_frame(d_path, self.image_size)
                out["gt_depth"] = d.unsqueeze(0).unsqueeze(0)

        if self.return_task_id:
            out["task_id"] = int(self.task_to_id[entry["task_name"]])

        return out


# ============================================================================
# Collate
# ============================================================================
def collate_fn(batch: List[Dict]) -> Dict:
    out: Dict = {
        "frames":    torch.stack([b["frames"] for b in batch], dim=0),
        "gs_params": [b["gs_params"] for b in batch],
        "text":      [b["text"] for b in batch],
    }
    if "gt_frames" in batch[0]:
        out["gt_frames"] = torch.stack([b["gt_frames"] for b in batch], dim=0)
    if "intrinsics" in batch[0]:
        out["intrinsics"] = torch.stack([b["intrinsics"] for b in batch], dim=0)
        out["extrinsics"] = torch.stack([b["extrinsics"] for b in batch], dim=0)
    if "gt_depth" in batch[0]:
        out["gt_depth"] = torch.stack([b["gt_depth"] for b in batch], dim=0)
    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([b["task_id"] for b in batch], dtype=torch.long)
    return out


# ============================================================================
# CLI smoke-test
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/manifest.json")
    ap.add_argument("--data_dir", default="outputs/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--n_gs_points", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ds = DatasetB(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        split=args.split,
        T=args.T,
        n_gs_points=args.n_gs_points,
        return_gt_frames=True,
        return_cameras=True,
        return_gt_depth=True,
        return_task_id=True,
    )
    print(f"Dataset[{args.split}]: {len(ds)} samples")

    s = ds[0]
    print("\n=== sample[0] ===")
    print(f"  frames     shape={tuple(s['frames'].shape)}  range=[{s['frames'].min():.3f}, {s['frames'].max():.3f}]")
    g = s["gs_params"]
    print(f"  gs.mu      shape={tuple(g.mu.shape)}  range=[{g.mu.min():.2f}, {g.mu.max():.2f}]")
    print(f"  gs.cov     shape={tuple(g.cov.shape)}")
    print(f"  gs.scale   shape={tuple(g.scale.shape)}  range=[{g.scale.min():.2f}, {g.scale.max():.2f}]")
    print(f"  gs.sh      shape={tuple(g.sh.shape)}")
    print(f"  gs.opacity shape={tuple(g.opacity.shape)}  range=[{g.opacity.min():.3f}, {g.opacity.max():.3f}]")
    print(f"  text       {s['text']!r}")
    print(f"  intrinsics shape={tuple(s['intrinsics'].shape)}")
    print(f"  extrinsics shape={tuple(s['extrinsics'].shape)}")
    if "gt_depth" in s:
        print(f"  gt_depth   shape={tuple(s['gt_depth'].shape)}  range=[{s['gt_depth'].min():.2f}, {s['gt_depth'].max():.2f}]m")
    print(f"  task_id    {s['task_id']}")

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)
    batch = next(iter(loader))
    print(f"\n=== batch (B={args.batch_size}) ===")
    print(f"  frames:    {tuple(batch['frames'].shape)}")
    print(f"  gs_params: list len={len(batch['gs_params'])}, "
          f"N per sample = {[len(g) for g in batch['gs_params']]}")
    print(f"  text:      {batch['text']}")
    print("\nOK ✓")
