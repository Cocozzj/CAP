# Ablation 总入口

> 三类消融 + 一键运行脚本。论文映射：PDF §5.2 / Act4D MD §3+§5.1 / Experiment.md §2.B + §3.1。

## 目录布局

```
eval/ablation/
├── ksweep/              # K-sweep + Theorem 1 验证（4 K + main K=512 = 5 点拟合）
├── module/              # Tier 1: 6 个模块消融（A→B 完整流程）
├── loss/                # Tier 2: 6 个 loss-term 消融（A only）
├── run_all.sh           # 一键串起来：K-sweep → module → loss
└── README.md            # 本文件
```

## 一键全量

```bash
cd /workspace/CAP
nohup bash eval/ablation/run_all.sh > ablation_master.log 2>&1 &
```

8×H100 上预计 **~86 GPU·h ≈ 3.6 天**。Resumable（每个 variant 检查 ckpt 是否已存在，跳过已完成的）。

## 单 phase 跑

如果想 cherry-pick 跑某个 phase：

```bash
# Phase 1: K-sweep（验证 Theorem 1，最便宜也最重要）
bash eval/ablation/ksweep/train_sweep.sh && \
bash eval/ablation/ksweep/eval_sweep.sh && \
python eval/ablation/ksweep/plot_theorem1.py \
    --summary runs/ablation/ksweep/_eval/summary.json \
    --output  runs/ablation/ksweep/_eval/theorem1.pdf

# Phase 2: Module（Tab 6 主消融）
bash eval/ablation/module/train_a.sh
bash eval/ablation/module/finetune_b.sh
bash eval/ablation/module/eval_all.sh
python eval/ablation/module/aggregate.py

# Phase 3: Loss（附录 Tab S1）
bash eval/ablation/loss/train_a.sh
bash eval/ablation/loss/eval_all.sh
python eval/ablation/loss/aggregate.py
```

## 总变体清单（19 个）

| Phase | 变体数 | 训练目标 | 论文输出 |
|---|---|---|---|
| ksweep | 4（不含 K=512 main）| A only | Fig 3 + d 的实测值 |
| module | 6 | A + B fine-tune | **Tab 6 主消融** |
| loss   | 6 | A only | Tab S1 附录 |
| **合计** | **16 个新 run** | | |

加上已有的 main 模型（3 seed × A + B），论文一共 ~19 个 run 进表。

## Trainer 侧依赖

我在 `train/trainer.py` 加了两个 ablation flag（已 commit 到本地 4DTokenizer/CAP/train/trainer.py）：

- `--no-physics`：所有 stage `enable_physics=False, enable_physics_loss=False`
- `--no-kl-anneal`：所有 stage `LossSpec.anneal_cvae_kl=False`

记得 sync trainer.py 到 server。

## 失败时怎么继续

整套流程是幂等的——任何 phase 中途崩了，重跑 `run_all.sh` 会跳过已 ckpt 的变体，从下一个继续。

如果想强制重跑某个 variant：
```bash
rm -rf runs/ablation/module/no_lipschitz/seed_0/ckpt/
VARIANTS=no_lipschitz bash eval/ablation/module/train_a.sh
```

## 加新变体

只改 `module/variants.py` 或 `loss/variants.py` 一个文件。bash 脚本自动 enumerate。

## 论文里的位置

- **Fig 3 (Theorem 1)**：`ksweep/_eval/theorem1.pdf` 直接进 paper §5.1
- **Tab 6 (主消融)**：`module/_aggregate/table6.md` 进 paper §5
- **Tab S1 (loss 细消融)**：`loss/_aggregate/table_loss.md` 进 appendix
- **Discussion**：从这三表里挑差距最大的几行写 ablation 分析（典型预期：no_algebraic 在 Δ_clos 上爆炸；no_physics 在 cross-material 上崩；no_L_comm 几乎不变 → 证明 commutator 是 soft prior）
