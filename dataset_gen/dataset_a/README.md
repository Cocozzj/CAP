# Dataset-A: Synthetic Multi-View Action Dataset

Synthetic dataset for the 4D Action Tokenizer paper. Built on
[**PartNet-Mobility**](https://sapien.ucsd.edu/) objects in
[**SAPIEN**](https://sapien.ucsd.edu/), with multi-view trajectory videos and a
static 3DGS primitive for each trajectory's starting state.

---

## 1. 整体 pipeline

```
[PartNet-Mobility 本地拷贝]
        │
   (1) 对象筛选 (motion saliency + 类别白名单)
        ▼
  object_list.json   (~150 个对象)
        │
   (2) Trajectory 规划 + 物理仿真 (SAPIEN)
        ▼
  trajectories.json  (~5000 条 joint 序列)
        │
   (3) 多视角同步渲染 (3 视角, 13s × 30fps = 390 帧)
        ▼
  data/<traj_id>/{front,side,high_oblique}.mp4
                 + cameras.json + physics.json + meta.json
        │
   (4) 每条 trajectory 第一帧 → 静态 3DGS (MVSplat 前向)
        ▼
  data/<traj_id>/init_gs.ply
        │
   (5) 划分 + 打包
        ▼
  splits.json + manifest.json
```

**核心设计原则**：

- **3DGS 是模型的输入**（起始场景），不是模型要学的目标。模型自己产生后续每帧的 4D 演化。
- **多视角 RGB 视频是监督信号**（reprojection loss）。模型预测的 4DGS 渲染回 3 视角，和这些视频比 loss。
- **物理参数 ρ**（friction/damping/mass）是合成数据独有的 bonus GT，给物理插件模块直接监督。
- **3DGS per-trajectory 而非 per-object**：每条 trajectory 第一帧的关节状态可能不同（close/push 任务起点是部分打开），所以每条 trajectory 都重建一次自己的初始 3DGS，避免做关节驱动动画化的复杂逻辑。

---

## 2. 数据规格（按 PDF）

| 项 | 值 |
|---|---|
| 对象类别 | 18（15 PartNet 刚体 + 2 程序化软体 + 1 Kettle）|
| 原子高层任务 | 8（open / close / pull / push / rotate / **squeeze / fold / pour**）|
| 2 步组合任务（in-train） | 4（open_close / close_open / pull_push / push_pull）|
| 3 步组合任务（eval-only） | 4（open_close_open / close_open_close / pull_push_pull / open_open_more）|
| 4 步组合任务（eval-only） | 3（open_close_open_close / close_open_close_open / pull_push_pull_push）|
| 刚体原子 trajectory（5 task）| ~4500（占总 atomic 90%）|
| 软体原子 trajectory（squeeze/fold/pour）| ~500（占总 atomic 10%，由 `soft_body_fraction` 控制）|
| 原子 trajectory 总数 | **~5000** |
| 2 步组合 trajectory 数（in-train） | ~1080 |
| 3 步组合 trajectory 数（eval-only） | ~228 |
| 4 步组合 trajectory 数（eval-only） | ~180 |
| **总 trajectory 数** | **~6800** |
| 每条 trajectory 视角数 | 3（前方 + 侧 50° + 高斜 -40°）|
| 每条 trajectory 时长 | 13 秒 |
| 帧率 | 30 fps（PDF 范围 20-60 fps）|
| 帧数 / 视频 | 390 |
| 渲染分辨率 | 256² 默认（512² 可选）|
| 总对齐序列数 | 6800 × 3 = 20400 |
| 总数据量（压缩后估计）| ~250 GB（视频 + 3DGS）|

---

## 3. 工具栈

| 用途 | 工具 | 备注 |
|---|---|---|
| 刚体物理引擎 + 渲染 | [SAPIEN](https://sapien.ucsd.edu/) | PartNet-Mobility 亲儿子，URDF 直接 load |
| 软体程序化生成 + 渲染 | [trimesh](https://trimsh.org/) + [pyrender](https://pyrender.readthedocs.io/) | 不依赖 PBD/MPM 物理仿真（路线 A） |
| 3DGS 前向重建 | [MVSplat](https://github.com/donydchen/mvsplat) | 3 视角 + GT 相机的最佳匹配 |
| 3DGS fallback | [gsplat](https://github.com/nerfstudio-project/gsplat) | per-scene 优化，慢但稳；当 MVSplat 在合成数据上效果不好时 fallback |
| 数据格式 | 目录 + JSON / WebDataset | 默认目录便于调试，可选 tar shards 便于分发 |

为什么不用 PyBullet：SAPIEN 直接 load PartNet-Mobility URDF，不需要适配；多相机 GPU 渲染更快；同 PartNet 团队维护。

为什么 MVSplat 而不是 VGGT/AnySplat：MVSplat 直接出 Gaussians（VGGT 出 point map 还要再转），且专为 2-4 posed views 设计（你的 3 视角 + GT 相机正合适）。AnySplat 较新，仓库稳定性不如 MVSplat。

---

## 4. 安装

```bash
# 装基础依赖
pip install -r requirements.txt

# 装 MVSplat（按 https://github.com/donydchen/mvsplat 的说明）
# 这一步会下载预训练权重
```

---

## 5. 运行

### 设置 PartNet-Mobility 路径

PartNet-Mobility 本地解压后的目录结构应该是：

```
/your/path/partnet_mobility_v0/
├── dataset/
│   ├── 100147/
│   │   ├── mobility.urdf
│   │   ├── mobility_v2.json
│   │   ├── meta.json
│   │   └── textured_objs/
│   ├── 100231/
│   └── ...
```

### 一键全跑

```bash
cd dataset/dataset_a
python scripts/run_all.py --partnet_root /your/path/partnet_mobility_v0
```

### 分步跑（推荐第一次用，方便调试）

```bash
# Step 1: 筛对象（30 分钟，可加 --no_saliency 加速）
python scripts/01_filter_objects.py \
    --partnet_root /your/path/partnet_mobility_v0 \
    --out outputs/object_list.json

# Step 2: 跑物理仿真生成 trajectory 关节序列（CPU，~1-2 小时）
python scripts/02_generate_trajectories.py \
    --object_list outputs/object_list.json \
    --out outputs/trajectories.json

# Step 3: 多视角同步渲染（GPU 密集，~6-12 小时 / 4 GPU 并行）
python scripts/03_render_multiview.py \
    --trajectories outputs/trajectories.json \
    --object_list outputs/object_list.json \
    --num_workers 4 \
    --out outputs/data/

# Step 4: 第一帧 → 3DGS（GPU，~1-3 小时 / 单 GPU）
python scripts/04_init_gs_from_first_frame.py \
    --data_dir outputs/data/ \
    --backend mvsplat \
    --fallback gsplat

# Step 5: 划分 + 打包
python scripts/05_split_and_pack.py \
    --trajectories outputs/trajectories.json \
    --data_dir outputs/data/ \
    --out_splits outputs/splits.json
```

> **注意**：脚本编号在新流程下重排了。如果你看到旧的 `02_reconstruct_static_gs.py`，那是旧 pipeline 的产物（per-object 静态重建），新流程不再使用，可删可留作 fallback。

---

## 6. 输出结构

```
outputs/
├── object_list.json           # ~150 对象 + saliency 评分
├── trajectories.json          # ~5000 条 joint 序列 + 物理参数
├── data/                      # 每条 trajectory 一个目录
│   ├── A_Door_100147_open_s001/
│   │   ├── front.mp4              # 256² @ 30fps × 390 帧 ≈ 13s
│   │   ├── side.mp4
│   │   ├── high_oblique.mp4
│   │   ├── init_gs.ply            # 第一帧的静态 3DGS（模型输入）
│   │   ├── cameras.json           # 3 相机的 K + extrinsics
│   │   ├── trajectory.npz         # joint_qpos[390], object_pose[390,7]
│   │   ├── physics.json           # {friction, damping, mass, ...}
│   │   └── meta.json              # traj_id, obj_id, category, task, ...
│   └── ... (5000 个目录)
├── splits.json                # train/val/test_iid + OOD splits + Dataset-D
└── manifest.json              # 全局索引：traj_id → split + 路径
```

### 单 sample 的存储量

| 项 | 大小 |
|---|---|
| 3 个 MP4 (CRF 23) | ~1-2 MB |
| init_gs.ply | ~10 MB |
| trajectory.npz | ~50 KB |
| 元数据 JSON | ~5 KB |
| **小计** | **~12 MB** |

5000 trajectory × ~12 MB ≈ **60 GB**（compresssed）。

---

## 7. 数据划分

| Split | 用途 | 内容 |
|---|---|---|
| `train` | 训练 | 原子 + 2 步组合 trajectory（in-train 部分）|
| `val` | 选超参 / early stop | 同上 |
| `test_iid` | 同分布评估 | 同上 |
| `test_ood_unseen_pair` | 零样本 (cat × task) 组合泛化 | (cat × task) 在 train 没出现过 |
| `test_ood_unseen_object` | 实例级泛化 | obj_id 在 train 没出现过 |
| **`test_compositional_long`** | **多步组合零样本泛化（PDF "组合长度 3 步" 曲线点）** | **3 步+ 组合 trajectory，模型从未见过链式 supervision** |
| `dataset_d_train` / `dataset_d_test` | 类别级泛化（独立 split）| 整个 category 留出 |

详细配置在 `configs/default.yaml#splits` 和 `configs/compositions.yaml`。

### 组合 trajectory 的设计

每条组合 trajectory 仍然是 13 秒，但 motion 阶段被切成 N 段：

```
[pre_settle 0.5-1.5s] [motion_1 ~3s] [inter_settle 0.3-0.8s] [motion_2 ~3s] [post_settle 3-5s]
```

3 步组合时 motion 段相应缩短到 ~2 秒/段。每段 motion 都是 min-jerk profile，连续段的边界条件确保 qpos 连续。

`meta.json` 里多出三个字段帮助下游使用：

```json
{
  "is_composition": true,
  "composition_steps": ["open", "close"],
  "sub_action_frame_ranges": [[45, 135], [165, 255]],
  "eval_only": false
}
```

`sub_action_frame_ranges` 给出每个 sub-action 的 motion 在 390 帧序列里的起止帧，方便：
- 训练时给每段 motion 单独打 sub-action 标签（hierarchical supervision）
- 评估时分别测每个 sub-action 的成功率

---

## 8. 训练时如何消费

DataLoader 的伪代码：

```python
class DatasetA(torch.utils.data.Dataset):
    def __init__(self, manifest_path, split):
        self.entries = [e for e in load_json(manifest_path)["entries"]
                        if e["split"] == split]

    def __getitem__(self, idx):
        e = self.entries[idx]
        traj_dir = self.root / e["rel_dir"]
        return {
            "init_gs":     load_ply(traj_dir / "init_gs.ply"),
            "rgb_views":   {cam: load_mp4(traj_dir / f"{cam}.mp4")
                           for cam in ["front", "side", "high_oblique"]},
            "cameras":     load_json(traj_dir / "cameras.json"),
            "physics":     load_json(traj_dir / "physics.json"),
            "action":      e["task_name"],
            "trajectory":  np.load(traj_dir / "trajectory.npz"),
        }
```

模型流程：

```
init_gs (输入) + 视频 → encoder → atomic action tokens
                                ↓
                              planner → task tokens
                                ↓
                            executor (作用于 init_gs) → 每帧 4DGS
                                ↓
                       渲染回 3 视角 → 与 rgb_views 比 reprojection loss
```

物理插件模块用 `physics` 做直接监督（mass/friction 等给 ρ-子空间）。

---

## 9. Pilot 测试（强烈建议第一次跑前先做）

不要直接 5000 条全跑，先 smoke test 1-3 个对象 × 1 个 task：

```bash
# 临时改 default.yaml: instances_per_category: 1, target_total_trajectories: 5
python scripts/run_all.py --partnet_root /your/path --out outputs_test/
```

检查：
- `outputs_test/data/<traj_id>/front.mp4` 能正常播放，物体在画面中，关节在动
- `init_gs.ply` 能在 3DGS viewer（如 SuperSplat）打开，几何看起来对
- `meta.json` / `physics.json` 字段完整

确认后再 scale up 到 5000 条。

---

## 10. 各步骤时间预估（4× A100 80GB）

| Step | 时长 | 说明 |
|---|---|---|
| 1. 对象筛选 | 30 min - 2 hr | motion saliency 渲染是瓶颈 |
| 2. Trajectory 规划 | 1-2 hr | CPU 跑物理 |
| 3. 多视角渲染 | 6-12 hr | GPU 渲染密集，4 worker 并行 |
| 4. 3DGS 前向（MVSplat）| 1-3 hr | 单 GPU 即可 |
| 5. 划分 + 打包 | 5 min | 纯 IO |
| **合计** | **~12-20 hr** | 一次性完成 |

---

## 11. 软体任务的设计选择

PDF 提到流体/软体动作（squeeze, fold, pour），但**没规定怎么实现**。我们走"路线 A"——**程序化形变**而非真实物理仿真：

| 任务 | 对象 | 实现 |
|---|---|---|
| **squeeze** | SoftToy（程序化生成的 cube/sphere）| 各向异性 scale（沿随机轴压缩到 40-70%，其他轴轻微 bulge） |
| **fold** | Cloth（程序化生成的平面 grid mesh） | 沿一条 hinge 线把一半旋转 120-180°，另一半静止 |
| **pour** | Kettle（PartNet 的铰接对象） | 不动 lid joint，**驱动 root pose 整体倾斜** 60-90° |

**为什么不用真实 PBD/MPM 物理仿真**：
- 模型只看 RGB 像素，看不出形变是真实物理还是参数化的
- 训练目标是学 action token，不是学物理仿真
- 路线 A 实现 1-2 天，路线 B（PyBullet 软体）3-4 天，路线 C（Warp DiffMPM）5-7 天
- 物理插件论文章节如果要"真实物理"卖点，可以后续升级 SqueezeTask/FoldTask 内部的 deformation 函数为 PBD 调用，trajectory schema 不变

**对 squeeze/fold 渲染**：单独走 pyrender 渲染路径（不经过 SAPIEN），因为 SAPIEN 不支持每帧更新可形变 mesh。每帧从 `soft_object_spec`（含 primitive 类型、size、color）重建 rest mesh，然后施加 `deformation_params_per_frame` 里的形变参数（scale 矩阵 / hinge 角），再渲染到 3 个相机。相机参数和 SAPIEN renderer 完全一致，保证多视角空间布局一致。

**对 pour 渲染**：pour 走 SAPIEN 渲染路径，但驱动 root pose 而不是 joint。`renderer.py` 检测 `object_type == "articulated_root_pose"` 时调用 `articulation.set_root_pose()` 而不是 `set_qpos()`。

## 12. 已知风险与应对

| 风险 | 概率 | 应对 |
|---|---|---|
| MVSplat 在 PartNet 合成数据上 OOD 表现差 | 中 | 切到 `--backend gsplat`（per-scene 优化，慢但稳） |
| SAPIEN 渲染在 headless 服务器需要 EGL/Vulkan 配置 | 高 | 用 SAPIEN 官方 docker 或 `xvfb-run` wrapper |
| 某些 PartNet 实例的关节失败仿真崩溃 | 中 | 已加 `success` 标记，失败的 trajectory 不进 split |
| 长 trajectory（13s × 390 帧）渲染单条时间过长 | 中 | 默认配置已平衡；想加速可调 `frames_per_trajectory` 到 200 |

---

## 13. 配置要点

所有 scale / quality 旋钮在 `configs/default.yaml`。常改的：

```yaml
scale:
  instances_per_category: 12       # 每类对象数
  trajectories_per_pair: 8         # 每个 (obj × task) 多少 trajectory
  target_total_trajectories: 5000  # 总数上限

render:
  resolution: 256                  # 256 训练快，512 reviewer 友好
  fps: 30
  duration_seconds: 13.0           # PDF 规格
  frames_per_trajectory: 390       # = fps * duration_seconds

gs:
  backend: mvsplat                 # mvsplat | gsplat | hybrid
  refine_iters_after_mvsplat: 0    # >0 则用 gsplat 短 refine 增强 MVSplat 输出

splits:
  ood_pair_fraction: 0.10
  held_out_categories: [Refrigerator, Faucet, Window]   # Dataset-D
```

---

## 14. Tests

```bash
pytest tests/                       # 不依赖 SAPIEN
SAPIEN_TEST=1 pytest tests/         # 完整测试
```

---

## 15. Future Work / 已知边界（PDF 提到但本数据集**未实现**）

下面是 PDF 提到、但我们当前 Dataset-A **没做**的部分。reviewer 问起就照这个表答辩。

| 缺的能力 | PDF 出处 | 为什么没做 | 升级路径 |
|---|---|---|---|
| **抓起 / 放下**（pick_up / put_down） | "动作类型包括…抓起/放下等" | 需要 SAPIEN 末端执行器 + 抓取仿真，~3 天工程 | SAPIEN 支持 panda gripper；新增 `GraspTask` 用 EE 控制接近物体 → 闭合手指 → 抬起 |
| **多对象跨场景任务**（"拿杯子放架子上"）| "复杂任务规划：从桌上拿起杯子放到架子上" | 需要多对象场景 + 抓取 + 放置目标判定，~1 周工程 | 在 SAPIEN 里搭多对象 scene；定义 `MultiObjectTask` 抽象 |
| **真实物理软体仿真**（PBD/MPM）| "对流体/软体采用 Position-Based Dynamics" | 我们用了路线 A（程序化形变），不是真实物理 | 把 `tasks/soft.py` 里的 `apply_anisotropic_scale` / `apply_hinge_fold` 替换成 Warp DiffPBD 调用，trajectory schema 不变 |
| **每帧 4DGS GT** | "可获取精确 3D GT，例如每帧高斯参数" | 与你确认过：模型自己产生 4D，数据集只提供 init_gs 第一帧 + 多视角 RGB 监督 | 不需要升级 |
| **跨材质 / 跨摩擦数据**（专门的 Dataset-C 子集）| "跨材料泛化…体积保持率…弹性恢复率" | physics_params 已记录 friction/damping/mass，但默认不做大范围扫参 | 改 `default.yaml#physics.randomize_*`，增加扫参范围；推荐为 Dataset-C 单独跑一份 |
| **真实视频背景 / 复杂光照** | （PDF 没明说，但 Sim-to-real 需要）| 当前白背景 + 单方向光，sim2real gap 大 | 加 domain randomization：光照 / 纹理 / 背景 hdri |

### 论文里如何 framing 这些"未做"

- 主结果建立在 PDF 严格 spec 上：15 类对象 × 8 task × 5000 段 × 3 视角 × 13s
- 抓取 + 多对象写到 Discussion → Future Work，标"超出本工作 scope"
- 软体物理写到 Method → Physics Plugin → "we adopt parametric deformation as our v1 implementation; the trajectory schema supports plug-in differentiable physics backends"

这样 reviewer 不会因为"PDF 里说了的没看到"而否定，而是会理解为有意识的 scope 选择。

## 16. 与 PDF 的对应关系

| PDF 描述 | 实现 |
|---|---|
| "PartNet-Mobility 中具有运动部件的对象" | `object_loader.py` 按 `mobility_v2.json` 过滤 |
| "PyBullet 物理引擎中设计脚本执行各种动作" | `tasks/*.py` 在 SAPIEN 里跑（PhysX 物理引擎，等价于 PyBullet）|
| "3 视角同步录制" | `renderer.py` 多相机 batched render |
| "13 秒 × 20-60 帧" | `default.yaml` 配 fps=30 → 390 帧 |
| "对象类别 15 种 × 任务动作 10 种 × 总视频 5000" | `configs/object_categories.yaml` + `configs/tasks.yaml` |
| "(对象类别 × 任务) 留出做零样本组合泛化" | `splitter.py` 的 `test_ood_unseen_pair` |
| "组合动作长度 2/3/4 步成功率曲线" | 2 步组合进 train，3 步组合进 `test_compositional_long`（eval-only） |
| "多视角监督（重投影损失）" | DataLoader 出 `rgb_views` + `cameras` 给训练用 |
| "精确 3D GT，例如每帧高斯参数" | `init_gs.ply`（仅第一帧）+ `trajectory.npz`（关节序列）|
| "真实 ρ 值"（物理参数）| `physics.json` |

PDF 没规定但我们做的设计选择：
- 用 **SAPIEN 替代 PyBullet**（兼容更好）
- 用 **MVSplat** 提取第一帧 3DGS（per-scene gsplat 作 fallback）
- 8 个 task 覆盖 PDF 提到的核心动作（5 刚体 + squeeze / fold / pour 软体）
- **软体走路线 A（程序化形变）** 而非真实 PBD/MPM
- 每对象 12 实例（PDF 没说具体数）
