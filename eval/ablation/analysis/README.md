# Ablation analysis (post-hoc, no GPU needed)

> 三个分析脚本，全部从已有训练 log + ckpt 中抽数据。**不重训、不重 eval、不占 GPU**。
> 跑完产出 paper §5 / appendix 用的 figure 4 + Tab S3 + Tab S5 等附加产物。

## 三个产物

| 脚本 | 输入 | 输出 | Paper 位置 |
|---|---|---|---|
| `extract_val_curves.py` | `runs/{module,loss}/<v>/seed_0/train_a.log` + `runs/main_a/seed_*/...log` | `val_curves.{png,csv,md}` | **Fig 4** |
| `per_stage_val_table.py` | 同上（grep 不同行）| `stage_val.{md,csv}` | **Tab S5** |
| `codebook_health.py` | 每个 variant 的 `main_exp_final.pt` 里的 codebook 张量 | `codebook_health.{md,csv}` | **Tab S3** |

## 一键跑

```bash
cd /workspace/CAP

mkdir -p runs/_analysis

python eval/ablation/analysis/extract_val_curves.py
python eval/ablation/analysis/per_stage_val_table.py
python eval/ablation/analysis/codebook_health.py
```

预计 ~1 min（log 解析快，codebook 是小张量、用 CPU 即可）。

## 输出结构

```
runs/_analysis/
├── val_curves.png           ← Fig 4: 9 个 variant 的 val_loss 曲线叠加
├── val_curves.csv           ← 每个 (variant, epoch) 对的原始数据
├── val_curves.md            ← 摘要表：best val + final val per variant
├── stage_val.md             ← Tab S5: 4 stage × 9 variant 的 best_val
├── stage_val.csv
├── codebook_health.md       ← Tab S3: codebook 健康度
└── codebook_health.csv
```

## 三张表/图各自能告诉 reviewer 什么

### Fig 4 — Val curves overlay

哪些 variant **早早就卡在某个高 val_loss plateau**（你之前发现 no_cvae / no_L_eq / no_L_hier 都收敛到 14.8），哪些跟 main 接轨（no_algebraic / no_hier）。

写法：
> *"Fig 4 shows that the four ablations whose validation loss diverges most sharply from main are precisely the ones that remove a CVAE / equivariance / hierarchical / commutator signal — confirming these terms carry non-redundant supervision."*

### Tab S5 — Per-stage val_loss

每个 ablation 在哪一个**curriculum stage** 开始 diverge：
- 如果 RIGID 阶段就差 → 表明该 loss 在最早期 perception 阶段就重要
- 如果 PHYSICS / FULL 才 diverge → 表明它在 fine-tune / 整体协调阶段才显出重要性

例如 no_physics 在前三 stage 跟 main 接近，到 FULL 才暴跌（因为 FULL 才大量调用物理仿真）。

### Tab S3 — Codebook utilisation

直接验证"代数结构是否被学到"的硬指标：
- main: 应该 ≥ 95% codes 都活
- no_algebraic / no_L_clos / no_L_inv: 可能 codebook 部分塌缩（去 group 约束 → 多余 code 没用）
- 这是**"我们的 group structure loss 真的让 codebook 利用率上升"** 的实证

写法：
> *"Removing algebraic constraints reduces effective codebook utilisation from X% to Y% (Tab S3) — direct evidence that the group-theoretic loss is what keeps the codebook from collapsing."*

## 跟 Tab 6 / Tab S1 (主表) 的关系

| 表/图 | 出处 | 状态 |
|---|---|---|
| Tab 6 | `eval/ablation/module/aggregate.py` (eval-driven) | 🟡 eval 数据有问题，待修 |
| Tab S1 | `eval/ablation/loss/aggregate.py` (eval-driven) | 🟡 同上 |
| **Fig 4 + Tab S3 + Tab S5** | **本目录脚本（log/ckpt-driven）** | ✅ **不依赖 eval**，可立即跑出 |

**即便 eval pipeline 还需要修，这三个产物已经够支撑 paper §5 + appendix 的关键叙事**——
- Fig 4 给 reviewer "ablation 训练动态"
- Tab S5 给 reviewer "stage-wise breakdown"
- Tab S3 给 reviewer "codebook 健康度"

如果 Tab 6 / Tab S1 最终还是出不来，这三张就**接管 §5 主体**。

## 加新分析的扩展点

想加更多分析？模板在每个脚本里很清晰：
- 解析 log → 用 `re` 抽行
- 读 ckpt → `torch.load(weights_only=False)` 然后 grep state_dict
- 生成 (csv, md) 双格式输出

例：想加"per-component loss bar chart" → 复制 `extract_val_curves.py`，把正则改成 `total=([\d.]+)` + 各 component grep，画 stacked bar。20 行代码。
