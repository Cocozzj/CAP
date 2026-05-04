"""PyTorch Dataset / DataLoader for Dataset-A.

Each `__getitem__(i)` returns a dict matching the model spec:

    {
        "frames":    Tensor [V, T, 3, H, W]   float32 in [0, 1]
        "gs_params": GSParameter (5 tensor fields)
        "text":      str                          # verb phrase
        # optional, gated by constructor flags:
        "gt_frames":  Tensor [V, T, 3, H, W]
        "intrinsics": Tensor [V, 3, 3]
        "extrinsics": Tensor [V, 4, 4]            (world -> cam)
        "task_id":    int
    }

Hard constraints enforced at __init__ / __getitem__:
    V >= 1            (we have cam0 / cam1 / cam2 = 3 cameras available)
    T >= 10 and T % 5 == 0    (default 30; resampled uniformly from 390 frames)
    H, W any (videos are 256x256; --image_size triggers resize)
    N >= 100         (we sample 10k points by default from the 50k init_gs.ply)
    frames in [0, 1] (uint8 -> /255)
    mu in WORLD frame (Step 4 mesh backend already produces world-frame points)
    mu / scale / opacity time-aligned to t=0 (init_gs.ply was built from first-frame qpos)

Use `from src.dataloader import DatasetA, collate_fn`.
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
# GSParameter
# ============================================================================
@dataclass
class GSParameter:
    """3DGS parameters at t=0 for one trajectory.

    N is variable across samples; same batch must use a List[GSParameter]
    in the collate. All other tensors stack normally.

        mu       [N, 3]      float32   world coords, meters; first-frame Gaussian centers
        cov      [N, 4]      float32   quaternion (w, x, y, z); model normalizes internally
        scale    [N, 3]      float32   log-scale per axis; linear = exp(scale), meters
        sh       [N, C_sh]   float32   SH coeffs (C_sh = gs_dim - 11; 48 for degree-3 RGB)
        opacity  [N, 1]      float32   in [0, 1]; ALREADY sigmoid'd, do NOT pass logit
    """

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
# task_name -> verb phrase
# ============================================================================
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
    "Box":                    "box",
    "Cloth":                  "cloth",
    "Dishwasher":             "dishwasher",
    "Door":                   "door",
    "Faucet":                 "faucet",
    "Kettle":                 "kettle",
    "Laptop":                 "laptop",
    "Microwave":              "microwave",
    "Oven":                   "oven",
    "Refrigerator":           "refrigerator",
    "Scissors":               "scissors",
    "SoftToy":                "soft toy",
    "StorageFurniture_Door":  "cabinet door",
    "StorageFurniture_Drawer": "drawer",
    "Suitcase":               "suitcase",
    "TrashCan":               "trash can",
    "Window":                 "window",
}


def task_to_text(task_name: str, obj_category: str) -> str:
    """Build a natural-language verb phrase from task + category.

    Atomic: 'open' + 'StorageFurniture_Drawer'  ->  "open the drawer"
    2-step: 'comp:open_close' + 'Door'          ->  "open then close the door"
    3-step: 'comp:open_close_open' + 'Door'     ->  "open, close, then open the door"
    Special: 'comp:open_open_more'              ->  "open, then open the X further"
    """
    cat_nl = _CATEGORY_NL.get(obj_category, obj_category.lower())
    if not task_name.startswith("comp:"):
        tpl = _VERB_TEMPLATES.get(task_name, task_name + " the {cat}")
        return tpl.format(cat=cat_nl)

    steps = task_name[len("comp:"):].split("_")
    # special-case the "open_open_more" pattern produced by the trajectory generator
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
        arr = vr.get_batch(list(frame_indices)).asnumpy()  # (T, H, W, 3) uint8
    except ImportError:
        import imageio
        reader = imageio.get_reader(str(mp4_path))
        try:
            arr = np.stack([reader.get_data(int(i)) for i in frame_indices], axis=0)
        finally:
            reader.close()

    if target_size is not None and arr.shape[1] != target_size:
        # PIL resize per frame (small T, fine to do in Python)
        from PIL import Image
        out = np.empty((arr.shape[0], target_size, target_size, 3), dtype=np.uint8)
        for t in range(arr.shape[0]):
            out[t] = np.array(
                Image.fromarray(arr[t]).resize((target_size, target_size),
                                                Image.BILINEAR)
            )
        arr = out

    arr = arr.astype(np.float32) / 255.0           # (T, H, W, 3)
    arr = np.transpose(arr, (0, 3, 1, 2))           # (T, 3, H, W)
    return torch.from_numpy(arr)


# ============================================================================
# PLY loading
# ============================================================================
# Standard 3DGS sigmoid normalization for SH degree-0 colors:
#   color = SH_DC * SH_C0 + 0.5  where SH_C0 = 0.28209479177387814
_SH_C0 = 0.28209479177387814


def load_init_gs_ply(ply_path: Path,
                     n_points: Optional[int] = None,
                     seed: int = 0,
                     sh_degree: int = 3) -> GSParameter:
    """Load init_gs.ply (Step 4 output) and convert to GSParameter.

    Handles two formats:
      (a) Mesh-backend PLY: has fields x/y/z + r/g/b + scale_x/y/z (linear) +
          rot_w/x/y/z + opacity (already in [0,1]).
      (b) Standard 3DGS PLY: x/y/z + f_dc_0..2 + f_rest_0..44 + opacity (logit) +
          scale_0..2 (log) + rot_0..3.
    """
    try:
        from plyfile import PlyData
    except ImportError as e:
        raise RuntimeError(
            "plyfile is required. Install: pip install plyfile"
        ) from e

    ply = PlyData.read(str(ply_path))
    # Use the underlying structured numpy array. PlyElement.dtype is a
    # method in newer plyfile versions, so we go through .data which is
    # always a numpy structured array with .dtype.names.
    v = ply["vertex"].data
    n = len(v)
    fields = set(v.dtype.names)

    # ---- mu (N, 3) ------------------------------------------------------
    mu = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    # ---- opacity (N, 1) ------------------------------------------------
    if "opacity" in fields:
        op = np.asarray(v["opacity"], dtype=np.float32)
    else:
        op = np.ones(n, dtype=np.float32)
    # If looks like logit (negative or > 1.5), apply sigmoid
    if op.min() < -0.05 or op.max() > 1.05:
        op = 1.0 / (1.0 + np.exp(-op))
    op = np.clip(op, 0.0, 1.0)
    opacity = op[:, None]

    # ---- scale (N, 3) — log-scale --------------------------------------
    if all(f"scale_{i}" in fields for i in range(3)):
        # standard 3DGS PLY: stored as log-scale already
        scale = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]],
                         axis=-1).astype(np.float32)
    elif {"scale_x", "scale_y", "scale_z"} <= fields:
        # mesh backend often writes linear scales under these names
        sx = np.asarray(v["scale_x"], dtype=np.float32)
        sy = np.asarray(v["scale_y"], dtype=np.float32)
        sz = np.asarray(v["scale_z"], dtype=np.float32)
        lin = np.stack([sx, sy, sz], axis=-1)
        # If max > 1.5, treat as linear; else assume already log
        if lin.max() > 1.5 or lin.min() >= 0.0:
            scale = np.log(np.maximum(lin, 1e-6))
        else:
            scale = lin
    else:
        # No scale stored — uniform tiny gaussians
        scale = np.full((n, 3), np.log(0.01), dtype=np.float32)

    # ---- cov / quaternion (N, 4) wxyz ----------------------------------
    if all(f"rot_{i}" in fields for i in range(4)):
        cov = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]],
                       axis=-1).astype(np.float32)
    elif {"rot_w", "rot_x", "rot_y", "rot_z"} <= fields:
        cov = np.stack([v["rot_w"], v["rot_x"], v["rot_y"], v["rot_z"]],
                       axis=-1).astype(np.float32)
    else:
        cov = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))

    # Normalize quaternion just in case
    qn = np.linalg.norm(cov, axis=-1, keepdims=True) + 1e-8
    cov = cov / qn

    # ---- SH coefficients (N, C_sh) -------------------------------------
    c_sh = (sh_degree + 1) ** 2 * 3   # 48 for degree 3
    sh = np.zeros((n, c_sh), dtype=np.float32)

    if all(f"f_dc_{i}" in fields for i in range(3)):
        # Standard 3DGS PLY: full SH in f_dc_0..2 + f_rest_0..44
        for i in range(3):
            sh[:, i] = np.asarray(v[f"f_dc_{i}"], dtype=np.float32)
        n_rest = c_sh - 3
        for i in range(n_rest):
            key = f"f_rest_{i}"
            if key in fields:
                sh[:, 3 + i] = np.asarray(v[key], dtype=np.float32)
    elif {"red", "green", "blue"} <= fields or {"r", "g", "b"} <= fields:
        # Mesh backend: only RGB color stored. Fill DC, leave higher orders 0.
        rk, gk, bk = ("red", "green", "blue") if "red" in fields else ("r", "g", "b")
        rgb = np.stack([np.asarray(v[rk]), np.asarray(v[gk]), np.asarray(v[bk])],
                       axis=-1).astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        # Inverse of standard SH-to-RGB mapping: sh_dc = (rgb - 0.5) / SH_C0
        sh[:, :3] = (rgb - 0.5) / _SH_C0

    # ---- subsample to n_points ----------------------------------------
    if n_points is not None and n > n_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n_points, replace=False)
        idx.sort()
        mu = mu[idx]; cov = cov[idx]; scale = scale[idx]
        sh = sh[idx]; opacity = opacity[idx]

    return GSParameter(
        mu=torch.from_numpy(np.ascontiguousarray(mu)),
        cov=torch.from_numpy(np.ascontiguousarray(cov)),
        scale=torch.from_numpy(np.ascontiguousarray(scale)),
        sh=torch.from_numpy(np.ascontiguousarray(sh)),
        opacity=torch.from_numpy(np.ascontiguousarray(opacity)),
    )


# ============================================================================
# Camera loading
# ============================================================================
def load_cameras(cameras_json_path: Path,
                 camera_names: Sequence[str]
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (intrinsics [V, 3, 3], extrinsics [V, 4, 4] world->cam, OpenCV)."""
    with open(cameras_json_path) as f:
        cams = json.load(f)
    Ks, w2cs = [], []
    for name in camera_names:
        c = cams[name]
        intr = c["intrinsics"]
        K = np.array([
            [intr["fx"], 0,           intr["cx"]],
            [0,           intr["fy"], intr["cy"]],
            [0,           0,           1],
        ], dtype=np.float32)
        w2c = np.array(c["extrinsics"]["world_to_camera_4x4"], dtype=np.float32)
        Ks.append(K)
        w2cs.append(w2c)
    return (torch.from_numpy(np.stack(Ks)),
            torch.from_numpy(np.stack(w2cs)))


# ============================================================================
# Dataset
# ============================================================================
class DatasetA(Dataset):
    """Dataset-A loader.

    Args:
        manifest_path: outputs/manifest.json from Step 5
        data_dir:      outputs/data (each subdir is one trajectory)
        split:         which split ('train' | 'val' | 'test_iid' |
                       'test_ood_unseen_pair' | 'test_ood_unseen_object' |
                       'test_compositional_long' | 'dataset_d_train' |
                       'dataset_d_test'). Filtered by membership in `splits` list.
        T:             frames per camera per sample (default 30)
        V_cameras:     which camera names to use (default all 3)
        n_gs_points:   subsample init_gs.ply to this many points (default 10000)
        image_size:    resize frames to this square size (default 256, no-op for our data)
        return_gt_frames: also return `gt_frames` (same as frames here; reconstruction loss)
        return_cameras:   also return `intrinsics` and `extrinsics`
        return_task_id:   also return integer `task_id` (sorted task_name index)
        require_text:     if False, return None for text (Stage 0/1); default True
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        data_dir: Union[str, Path],
        split: str = "train",
        T: int = 30,
        V_cameras: Sequence[str] = ("cam0", "cam1", "cam2"),
        n_gs_points: int = 10000,
        image_size: int = 256,
        return_gt_frames: bool = False,
        return_cameras: bool = False,
        return_task_id: bool = False,
        require_text: bool = True,
        sh_degree: int = 3,
        seed: int = 0,
    ):
        # Hard constraints
        if T < 10 or T % 5 != 0:
            raise ValueError(f"T must be >= 10 and a multiple of 5; got T={T}")
        if len(V_cameras) < 1:
            raise ValueError("Need at least one camera in V_cameras")
        if n_gs_points < 100:
            raise ValueError(f"n_gs_points={n_gs_points} too few; recommend >= 1000")

        self.data_dir = Path(data_dir)
        self.T = T
        self.V_cameras = list(V_cameras)
        self.n_gs_points = n_gs_points
        self.image_size = image_size
        self.return_gt_frames = return_gt_frames
        self.return_cameras = return_cameras
        self.return_task_id = return_task_id
        self.require_text = require_text
        self.sh_degree = sh_degree
        self.seed = seed
        self.split = split

        # Load + filter manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.entries = [e for e in manifest["entries"]
                        if split in e.get("splits", [])]
        if not self.entries:
            raise ValueError(
                f"No entries for split={split!r}. Available splits in this "
                f"manifest: {sorted({s for e in manifest['entries'] for s in e.get('splits', [])})}"
            )

        # Stable task_id mapping if requested
        if return_task_id:
            unique_tasks = sorted(set(e["task_name"] for e in self.entries))
            self.task_to_id = {t: i for i, t in enumerate(unique_tasks)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict:
        entry = self.entries[i]
        traj_dir = self.data_dir / entry["rel_dir"]
        n_total = int(entry.get("n_frames", 390))

        # Uniform frame sampling over the full trajectory
        frame_indices = np.linspace(0, n_total - 1, self.T, dtype=int).tolist()

        # Load frames per camera -> (V, T, 3, H, W)
        frames_per_cam = [
            _load_video_frames(traj_dir / f"{cam}.mp4",
                               frame_indices,
                               target_size=self.image_size)
            for cam in self.V_cameras
        ]
        frames = torch.stack(frames_per_cam, dim=0)
        # Sanity: enforce [0, 1] range
        frames = frames.clamp(0.0, 1.0)

        # Load GS init at t=0
        gs = load_init_gs_ply(
            traj_dir / "init_gs.ply",
            n_points=self.n_gs_points,
            seed=self.seed + i,
            sh_degree=self.sh_degree,
        )

        # text
        if self.require_text:
            text = task_to_text(entry["task_name"], entry["obj_category"])
        else:
            text = ""

        out: Dict = {
            "frames":    frames,
            "gs_params": gs,
            "text":      text,
        }

        if self.return_gt_frames:
            # By default GT == observed; subclass to return future-window frames.
            out["gt_frames"] = frames

        if self.return_cameras:
            K, w2c = load_cameras(traj_dir / "cameras.json", self.V_cameras)
            out["intrinsics"] = K
            out["extrinsics"] = w2c

        if self.return_task_id:
            out["task_id"] = int(self.task_to_id[entry["task_name"]])

        return out


# ============================================================================
# Collate
# ============================================================================
def collate_fn(batch: List[Dict]) -> Dict:
    """DataLoader collate. Stacks fixed-shape tensors; keeps per-sample
    GSParameter list (N varies)."""
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
    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([b["task_id"] for b in batch],
                                      dtype=torch.long)
    return out


# ============================================================================
# CLI smoke-test
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Smoke-test the Dataset-A loader.")
    ap.add_argument("--manifest", default="outputs/manifest.json")
    ap.add_argument("--data_dir", default="outputs/data")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--n_gs_points", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ds = DatasetA(
        manifest_path=args.manifest,
        data_dir=args.data_dir,
        split=args.split,
        T=args.T,
        V_cameras=("cam0", "cam1", "cam2"),
        n_gs_points=args.n_gs_points,
        return_gt_frames=True,
        return_cameras=True,
        return_task_id=True,
    )
    print(f"Dataset[{args.split}]: {len(ds)} samples")

    s = ds[0]
    print("\n=== sample[0] ===")
    print(f"  frames     shape={tuple(s['frames'].shape)}  "
          f"dtype={s['frames'].dtype}  "
          f"range=[{s['frames'].min():.3f}, {s['frames'].max():.3f}]")
    g = s["gs_params"]
    print(f"  gs.mu      shape={tuple(g.mu.shape)}        "
          f"range=[{g.mu.min():.2f}, {g.mu.max():.2f}]")
    print(f"  gs.cov     shape={tuple(g.cov.shape)}    (quat wxyz, normalized)")
    print(f"  gs.scale   shape={tuple(g.scale.shape)}     "
          f"log-scale, range=[{g.scale.min():.2f}, {g.scale.max():.2f}]")
    print(f"  gs.sh      shape={tuple(g.sh.shape)}     C_sh={g.sh.shape[1]}")
    print(f"  gs.opacity shape={tuple(g.opacity.shape)}      "
          f"range=[{g.opacity.min():.3f}, {g.opacity.max():.3f}]")
    print(f"  text       {s['text']!r}")
    print(f"  intrinsics shape={tuple(s['intrinsics'].shape)}")
    print(f"  extrinsics shape={tuple(s['extrinsics'].shape)}")
    print(f"  task_id    {s['task_id']}")

    from torch.utils.data import DataLoader
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    batch = next(iter(loader))
    print(f"\n=== batch (B={args.batch_size}) ===")
    print(f"  frames:    {tuple(batch['frames'].shape)}")
    print(f"  gs_params: list len={len(batch['gs_params'])}, "
          f"N per sample = {[len(g) for g in batch['gs_params']]}")
    print(f"  text:      {batch['text']}")
    if "gt_frames" in batch:
        print(f"  gt_frames: {tuple(batch['gt_frames'].shape)}")
    if "intrinsics" in batch:
        print(f"  intrinsics: {tuple(batch['intrinsics'].shape)}")
        print(f"  extrinsics: {tuple(batch['extrinsics'].shape)}")
    if "task_id" in batch:
        print(f"  task_id:   {batch['task_id'].tolist()}")
    print("\nOK ✓")
