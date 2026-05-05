# Dataset-B: Real-World Single-View Action Videos

Companion to Dataset-A. While Dataset-A is **simulated multi-view** (PartNet-Mobility + SAPIEN), Dataset-B is **real-world single-view** video, used to test how the 4D Action Tokenizer transfers to natural footage and uncontrolled object diversity.

## Per the project spec (修改后方案.pdf §5.1)

> Dataset-B (真实,单视角): 收集或使用公开动态视频数据集,强调多样性和复杂背景。可以采用 Something-Something V2 数据集中筛选出单对象交互片段 ... 以及 MPII Cooking Dataset 等具有明确动作的厨房视频。为了获得 3D 信息,可借助深度估计和光流方法作为辅助 ... Dataset-B 包含约 **1000 段视频**,涵盖 **8 类高层任务**。

## Comparison with Dataset-A

| Axis | Dataset-A | Dataset-B |
|---|---|---|
| Source | PartNet-Mobility + SAPIEN PhysX | Real-world video (SSv2 + MPII + optional Kinect) |
| Views | 3 cameras (cam0/1/2) | **1 camera (cam0)** |
| 3D ground truth | Exact (URDF FK) | **Estimated (MiDaS / DepthAnything)** |
| Action labels | Task definition + full qpos trace | **Weak verb phrase only** |
| Scale | 3,238 trajectories | ~1,000 clips |
| Role in pipeline | Stage-1 pretrain (geometry-accurate) | Stage-2 finetune (real-world transfer) |

## Verb vocabulary (matched to Dataset-A)

We restrict Dataset-B to the same 8 atomic verbs Dataset-A covers:
`open`, `close`, `pull`, `push`, `rotate`, `squeeze`, `fold`, `pour`.

This makes Stage-1 → Stage-2 transfer meaningful (the action codebook learned in Dataset-A pretraining can be directly probed on Dataset-B).

## Pipeline

```
[raw datasets] ──Step 1── [curated clip list]
                           │
                Step 2:    │ map class names → 8 atomic verbs + verb-phrase
                           ▼
                          [verb-mapped clips]
                           │
                Step 3:    │ trim/resize to 256² × 30fps × T frames
                           ▼
                          [standardized mp4s]
                           │
                Step 4:    │ DepthAnything v2 per-frame depth estimation
                           ▼
                          [depth.npz]
                           │
                Step 5:    │ back-project first-frame RGB+depth into GS PLY
                           ▼
                          [init_gs.ply]
                           │
                Step 6:    │ split (train/val/test) + write manifest
                           ▼
                          [trajectories.json, manifest.json, splits.json]
```

## Output layout (mirrors Dataset-A for shared dataloader compatibility)

```
outputs/
├── trajectories.json        # all clip metadata
├── manifest.json
├── splits.json
└── data/
    └── B_<source>_<clip_id>_<verb>/
        ├── cam0.mp4         # 256² × T frames × 30fps
        ├── depth.npz        # (T, 256, 256) float32 monocular depth
        ├── cameras.json     # estimated K (default fov 60°), no extrinsics (single view)
        ├── meta.json        # source, original_label, task_name, n_frames, ...
        └── init_gs.ply      # 10000 points back-projected from depth at t=0
```

## Status

- [ ] Step 1: source curation
- [ ] Step 2: verb mapping
- [ ] Step 3: clip standardization
- [ ] Step 4: depth estimation
- [ ] Step 5: GS init from depth
- [ ] Step 6: split & pack
- [ ] Dataloader matching `src/dataloader.py` interface from Dataset-A
