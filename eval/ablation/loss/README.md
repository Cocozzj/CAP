# Loss-term ablations (Tier 2, A only)

> **5 个 theorem-aligned 变体**（每个对应论文一个定理/命题），独立消每个 lambda。
> 仅在 Dataset-A 上训 + eval。
> 论文映射：PDF §5.2 / Act4D MD §3.B + §5.1 / Experiment.md §2.B Tier 2。

## 5 个 theorem-aligned 变体（默认）

| 变体名 | 对应理论 | 期望发现 | yaml 改动 |
|---|---|---|---|
| `no_L_clos` | **Theorem 1**（闭包误差上界）| closure_gap 显著上升，验证 L_clos 控制 K^{-1/d} scaling | `lambda_clos = 0` |
| `no_L_inv` | **Theorem 2**（逆元一致性）| inverse_gap 显著上升 | `lambda_inv = 0` |
| `no_L_eq` | **Proposition 3**（跨对象等变）| 跨对象 transfer accuracy 下降 | `lambda_eq = lambda_eq_cross = 0` |
| `no_L_hier` | **Proposition 4**（层级代数误差）| task↔atomic 解码不齐 | `lambda_hier = 0` |
| `no_L_nce` | **Proposition 5**（语法语义一致）| text-conditional 生成正确率下降 | `lambda_nce = 0` |

每个变体把对应的 lambda 置零，**其他 loss 保留** —— 这是关键，让我们能识别每个 loss 的**唯一边际贡献**。

## 全部 7 个变体（剩 2 个非 theorem 项，按需启用）

| 变体名 | 移除什么 | 默认跑？ |
|---|---|---|
| `no_L_comm` | commutator loss + 其 anneal | ⏸️ 默认不跑（commutator 在论文里是 "soft prior"，非定理项）|
| `no_kl_anneal` | CVAE β 退火（保持固定值）| ⏸️ 默认不跑（CVAE 训练细节，附录嫌琐碎）|

跑全部 7 个：

```bash
VARIANTS="no_L_clos no_L_inv no_L_eq no_L_hier no_L_nce no_L_comm no_kl_anneal" \
    bash eval/ablation/loss/train_a.sh
```

## 文件清单（结构同 module/）

| 文件 | 作用 |
|---|---|
| `variants.py`     | 7 个变体的 single source of truth（5 theorem-aligned + 2 misc）|
| `make_config.py`  | base config 按变体 patch 到 `configs/_ablation_loss/<variant>/` |
| `train_a.sh`      | A 训练循环（默认跑 5 个 theorem-aligned）|
| `eval_all.sh`     | A test 上跑 4 项 eval |
| `aggregate.py`    | 输出 `runs/loss/_aggregate/table_loss.{csv,md}` |

## 在 8×H100 上的执行流程

```bash
cd /workspace/CAP

# 1) 训 5 个 theorem-aligned 变体 (~5.3 h)
bash eval/ablation/loss/train_a.sh

# 2) 跑 eval (~5 × 8 min ≈ 40 min)
bash eval/ablation/loss/eval_all.sh

# 3) 聚合
python eval/ablation/loss/aggregate.py
```

## 总成本（默认 5 变体 × 75 ep）

| 阶段 | 单变体 | × 5 | 总 |
|---|---|---|---|
| Train A (75 ep) | ~64 min | 5.3 h | **5.3 h** |
| Eval | ~8 min | 40 min | **0.7 h** |
| **TOTAL** | | | **~6 GPU·h** |

## 跟 Tier 1 (module/) 的关系

Tier 1 `no_algebraic` = Tier 2 `no_L_clos + no_L_inv + no_L_eq + no_L_comm` 同时关。

- **Tier 1 Tab 6 (主表)**：粗粒度——一组 loss 一起关，看全套 algebraic structure 的整体贡献
- **Tier 2 Tab S1 (附录)**：细粒度——每个 loss 单独关，逐项匹配论文里的 5 个定理

paper 里 §5.1 (理论验证段) 直接引用 Tab S1：
> *"Table S1 confirms that each of the five theoretical claims is empirically grounded: removing L_clos / L_inv / L_eq / L_hier / L_nce sharply increases the corresponding metric while leaving the others largely unchanged."*

## 加变体怎么扩

编辑 `variants.py` 加 dict entry。其他文件不动。
