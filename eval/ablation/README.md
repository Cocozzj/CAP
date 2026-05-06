# Ablation 总入口

> 三类消融 + 一键运行脚本。**全部基于 Dataset-A**，不在 B 上 fine-tune。
> 论文映射：PDF §5.2 / Act4D MD §3+§5.1 / Experiment.md §2.B + §3.1。

## 目录布局

```
eval/ablation/
├── ksweep/              # K-sweep + Theorem 1 验证（4 K + main K=512 = 5 点拟合）
├── module/              # Tier 1: 6 个模块消融（A only）
├── loss/                # Tier 2: 6 个 loss-term 消融（A only）
├── run_all.sh           # 一键串起来：K-sweep → module → loss
└── README.md            # 本文件
```

## 一键全量

```bash
cd /workspace/CAP
nohup bash eval/ablation/run_all.sh > runs/ablation_master.log 2>&1 &
```

8×H100 上预计 **~24 GPU·h ≈ 1 天**（100 ep ablation × {4 K + 6 module + 6 loss}，smoke 实测 ~48s/ep 平均）。Resumable（每个 variant 检查 ckpt 是否已存在，跳过已完成的）。

## 单 phase 跑

如果想 cherry-pick 跑某个 phase：

```bash
# Phase 1: K-sweep（验证 Theorem 1，最便宜也最重要）
bash eval/ablation/ksweep/train_sweep.sh && \
bash eval/ablation/ksweep/eval_sweep.sh && \
python eval/ablation/ksweep/plot_theorem1.py \
    --summary runs/ksweep/_eval/summary.json \
    --output  runs/ksweep/_eval/theorem1.pdf

# Phase 2: Module（Tab 6 主消融，A only）
bash eval/ablation/module/train_a.sh
bash eval/ablation/module/eval_all.sh
python eval/ablation/module/aggregate.py

# Phase 3: Loss（附录 Tab S1，A only）
bash eval/ablation/loss/train_a.sh
bash eval/ablation/loss/eval_all.sh
python eval/ablation/loss/aggregate.py
```

## 总变体清单（16 个新 run）

| Phase | 变体数 | 训练目标 | 论文输出 |
|---|---|---|---|
| ksweep | 4 (K=64/128/256/1024，K=512 复用 main，K=2048 跳过)| A only, **100 ep** | Fig 3 + d 的实测值 |
| module | 6 | A only, **100 ep** | **Tab 6 主消融** |
| loss   | 6 | A only, **100 ep** | Tab S1 附录 |
| **合计** | **16** | | |

**注意**：ablation 100 ep < main 150 ep（节省 ~33% wallclock）。默认用 `STAGE_EPOCHS="25 20 20 35"` 显式分配每个 stage：

| Stage | 原 ep (main) | ablation (100 ep 总) | 比例 |
|---|---|---|---|
| RIGID | 35 | 25 | -29% |
| PLANNER | 35 | 20 | -43% |
| PHYSICS | 25 | 20 | -20% |
| FULL | 55 | **35** | -36% |

特意保留 FULL 仍是最大的 stage，因为它是模型整体 fine-tune 的关键阶段。

**改 epoch 预算的方式**：
- 跑 full 150 ep（跟 main 完全对称）：`STAGE_EPOCHS= MAX_EPOCHS= bash ...`
- 改其他比例：`STAGE_EPOCHS="20 20 20 40" bash ...`
- 改成 uniform cap：`STAGE_EPOCHS= MAX_EPOCHS=20 bash ...`

加上已有的 main 模型（3 seed × A，B finetune 不参与 ablation 比较），论文一共 ~17 个 run 进表。

## 为什么全部 A only

- Dataset-A 提供精确 GT（closure / inverse / commutator 解析定义）；B 用 MiDaS 伪深度，相对噪声大
- 消融对象是架构 / loss 设计，dataset-agnostic
- A-only 是 NeurIPS 消融的标准做法（类似 ImageNet-only ablation）
- 省 ~4 GPU·h，结构更对称

写 paper 时正文 §5 可以加一句脚注：
> *"All ablations trained on Dataset-A under identical curriculum and 1 seed. We focus on the synthetic setting because (i) Dataset-A provides exact GT for closure / inverse / commutator metrics, and (ii) the ablations target architectural / loss design, which is dataset-agnostic."*

## Trainer 侧依赖

`train/trainer.py` 加了两个 ablation flag：

- `--no-physics`：所有 stage `enable_physics=False, enable_physics_loss=False`
- `--no-kl-anneal`：所有 stage `LossSpec.anneal_cvae_kl=False`

记得 sync trainer.py 到 server。

## 失败时怎么继续

整套流程是幂等的——任何 phase 中途崩了，重跑 `run_all.sh` 会跳过已 ckpt 的变体，从下一个继续。

如果想强制重跑某个 variant：
```bash
rm -rf runs/module/no_lipschitz/seed_0/ckpt/
VARIANTS=no_lipschitz bash eval/ablation/module/train_a.sh
```

## 加新变体

只改 `module/variants.py` 或 `loss/variants.py` 一个文件。bash 脚本自动 enumerate。

## 论文里的位置

- **Fig 3 (Theorem 1)**：`runs/ksweep/_eval/theorem1.pdf` 直接进 paper §5.1
- **Tab 6 (主消融)**：`runs/module/_aggregate/table6.md` 进 paper §5
- **Tab S1 (loss 细消融)**：`runs/loss/_aggregate/table_loss.md` 进 appendix
- **Discussion**：从这三表里挑差距最大的几行写 ablation 分析（典型预期：`no_algebraic` 在 Δ_clos 上爆炸；`no_physics` 在 trajectory 上明显差；`no_L_comm` 几乎不变 → 证明 commutator 是 soft prior）
