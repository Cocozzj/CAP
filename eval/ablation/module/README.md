# Module-level ablations (Tier 1)

> 6 个模块消融，**只在 Dataset-A 上跑**，最后聚合成 Tab 6。
> 论文映射：PDF §5.2 / Act4D MD §3.A / Experiment.md §2.B Tier 1。

## 6 个变体（在 `variants.py` 单点定义）

| 变体名 | 移除什么 | 主要 yaml/CLI 改动 |
|---|---|---|
| `no_hier` | TaskTokenizer / 任务码本 | `planner.use_task_token: false` |
| `no_algebraic` | clos / inv / eq / comm 全部 loss | 5 个 lambda 置 0 |
| `no_cvae` | CVAE 训练 + 采样多样性 | `deterministic: true` + cvae lambda 置 0 |
| `no_physics` | 整个物理插件（DeformSim）| `--no-physics` CLI（trainer.py 新增）|
| `no_equivariance` | SE(3) 等变 loss | `lambda_eq = lambda_eq_cross = 0` |
| `no_lipschitz` | spectral-norm 正则的 loss 项 | `lambda_lip = 0` |

每个变体 A 训 **80 epoch**（默认 `MAX_EPOCHS=80`，main 是 150 ep）。**不在 B 上 fine-tune**——消融对象是架构 / loss 设计，dataset-agnostic。

## 文件清单

| 文件 | 作用 |
|---|---|
| `variants.py` | 6 个变体的 single source of truth（重要！加新变体只改这一个文件）|
| `make_config.py` | 把 base config + loss 按变体 patch，存到 `configs/_ablation_module/<variant>/` |
| `train_a.sh` | A 训练循环 |
| `eval_all.sh` | 跑 algebraic_gaps / trajectory / success / diversity 全套（A only）|
| `aggregate.py` | 聚合 eval JSON → Tab 6 csv + markdown |
| `finetune_b.sh` | （未启用）保留作为 escape hatch；run_all.sh 不调用 |

## Trainer 侧的依赖

`train/trainer.py` 加了两个 CLI flag：

- `--no-physics`：把每个 stage 的 `enable_physics=False, enable_physics_loss=False`
- `--no-kl-anneal`：把每个 stage 的 LossSpec `anneal_cvae_kl=False`

`make_config.py` 会把这些 flag 写到 `trainer_flags.txt`，bash 脚本读取并加到 torchrun。

## 在 8×H100 上的执行流程

```bash
cd /workspace/CAP

# 1) 训练 6 个 A 变体 (~30 GPU·h on 8×H100)
bash eval/ablation/module/train_a.sh

# 2) 跑全套 eval (~6 × 8 min ≈ 1 h，含 main 3-seed 对照)
bash eval/ablation/module/eval_all.sh

# 3) 聚合成 Tab 6
python eval/ablation/module/aggregate.py
#   → runs/module/_aggregate/table6.{csv,md}
```

## 单变体测试

跑全部之前先用单个变体短 epoch 验证 pipeline：

```bash
VARIANTS="no_lipschitz" MAX_EPOCHS=2 \
    bash eval/ablation/module/train_a.sh
```

跑通了说明 `make_config.py` patch 正确、trainer 接受新 flag、输出路径 OK。

## 总成本估算（8×H100 sequential）

| 阶段 | 单 variant | × 6 variants | 总 |
|---|---|---|---|
| Train A (80 ep, 默认) | ~2.7 h | 16 h | **16 h** |
| Eval (4 项 × A) | ~10 min | 1 h | **1 h** |
| **TOTAL** | | | **~17 GPU·h ≈ 0.7 wall-clock 天** |

并行说一下：6 个 A train 互不依赖，可以**多机并行**或**1×H100/seed 跑 6 个**（batch 减小到 2-4）。但 sequential 更稳，1.3 天等得起。

## 加变体怎么扩

1. 编辑 `variants.py`，新加一个 dict entry
2. 别的文件都不用动；脚本会自动 enumerate

## 失败时怎么诊断

| 症状 | 诊断 |
|---|---|
| `make_config.py: KeyError on dotpath` | yaml schema 改了；更新 variants.py 里的键路径 |
| trainer 启动报 `unrecognized arguments: --no-physics` | trainer.py 没 sync；server 上还是老版本 |
| L_Lip 在 no_lipschitz 上仍非零 | 看 TB；`lambda_lip=0` 只去掉 loss 项，spectral_norm wrap 仍在 vfield 里 |
| eval 全 `-`（aggregate 表里）| 看 `runs/module/<v>/seed_0/eval_a/<eval>/` 目录是否存在；可能 eval 脚本崩了 |

## 论文里的位置

- **Table 6 (主消融)**：`aggregate.py` 输出的 table6.md 直接进 paper §5 主表
- **Appendix**：每个变体的 train log + eval JSON 存档
- **Discussion**：从 Tab 6 哪几行差距最大反推贡献最大的设计点（典型预期：`no_algebraic` 应该在 Δ_clos 上爆炸；`no_physics` 在 trajectory metrics 上明显差）

## 写 paper 脚注的标准说辞

> All ablations trained on Dataset-A under identical curriculum and 1 seed;
> we focus on the synthetic setting because (i) Dataset-A provides exact GT
> for closure / inverse / commutator metrics, and (ii) the ablations target
> architectural / loss design which is dataset-agnostic. The main row in
> Tab 6 reports mean ± std over 3 seeds.
