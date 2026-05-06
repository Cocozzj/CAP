# K-sweep ablation (验证 Theorem 1)

> Closure error 应满足 `err ≈ A · K^(-1/d)`，其中 K 是 atomic codebook 大小、d 是 action manifold 内蕴维度。
> 跑一组不同 K 的训练 → 计算每个 ckpt 的 closure / inverse / commutator gap → log-log 拟合 → 出图。

## 文件清单

| 文件 | 作用 |
|---|---|
| `make_config.py` | 从 base `configs/config.yaml` 生成指定 K 的 config 副本 |
| `train_sweep.sh` | 对 K ∈ {64, 128, 256, 1024, 2048} 顺序跑 A 训练（K=512 是已有 main 模型，免训）|
| `eval_sweep.sh` | 把所有 K-ckpts 喂给 `eval.k_scaling_sweep` 聚合 closure/inverse/comm gaps |
| `plot_theorem1.py` | 把 sweep 结果画成对数-对数曲线 + `K^(-1/d)` 理论线，输出 NeurIPS 风格 PDF |

## 假设的目录布局

脚本假设你**从 `/workspace/CAP/` 运行**（和 trainer 一致）。把整个 `eval/ablation/ksweep/` 同步到那儿即可：

```
/workspace/CAP/
├── configs/                        # 已有
├── eval/                           # 已有 (k_scaling_sweep.py 等)
├── eval/ablation/ksweep/           # ← 本目录
├── runs/main_a/seed_0/...          # 已有 K=512 main 模型（3 seeds）
├── runs/finetune_b/seed_0/...      # 已有 B fine-tune
└── runs/ksweep/K{N}/               # ← sweep 输出（与 main_a 同级）
```

## 使用流程

```bash
cd /workspace/CAP

# 1) Train 4 个 K-variants（K=512 复用 main_a/seed_0）
bash eval/ablation/ksweep/train_sweep.sh

# 2) 跑 eval，聚合 gaps
bash eval/ablation/ksweep/eval_sweep.sh

# 3) 出图
python eval/ablation/ksweep/plot_theorem1.py \
    --summary runs/ksweep/_eval/summary.json \
    --output  runs/ksweep/_eval/theorem1.pdf
```

## 配置选择

- **默认 K = {64, 128, 256, 1024}**（4 个新点 + 已有 K=512 = **5 个拟合点**；K=2048 跳过）
- **每个 K 跑 1 seed**（按方案"主表 3 seeds，其他 1 seed"惯例）
- **80 epoch**（默认，main 是 150 ep）；想跑全 150 ep 用 `MAX_EPOCHS=150 bash ...`
- **8 GPU per run**。Sequential 总成本 ~4 × ~2.7 h ≈ **~12 GPU·hour**（约 0.5 GPU·day）

## 期望输出（论文素材）

- `runs/ksweep/_eval/summary.json` — per-K 的 closure / inverse / commutator gap
- `runs/ksweep/_eval/points.csv` — 表格用
- `runs/ksweep/_eval/theorem1.pdf` — Fig. 3 候选图（log-log，K^(-1/d) 理论线叠加）
- `runs/ksweep/_eval/fit.json` — 拟合的 (A, d, R²)

## 论文映射

PDF §5.2 实验五"消融实验"："**码本大小变化**：图 plot of error vs complexity 曲线"
+ Act4D MD §5.1：Theorem 1 经验验证
+ Experiment.md §3 Tier-1 K-scaling sweep
