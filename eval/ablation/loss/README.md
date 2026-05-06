# Loss-term ablations (Tier 2, A only)

> 6 个 loss 项消融，每个独立 zero-out 一个 lambda。
> 仅在 Dataset-A 上训 + eval（按方案的 Tier 2 预算）。
> 论文映射：PDF §5.2 / Act4D MD §3.B / Experiment.md §2.B Tier 2。

## 6 个变体（在 `variants.py` 单点定义）

| 变体名 | 移除什么 | yaml 改动 |
|---|---|---|
| `no_L_clos` | closure loss 单消 | `lambda_clos = 0` |
| `no_L_inv`  | inverse loss 单消 | `lambda_inv = 0` |
| `no_L_comm` | commutator + 其 anneal | `lambda_comm = lambda_comm_max = 0` |
| `no_L_hier` | 层级一致性 loss | `lambda_hier = 0` |
| `no_L_nce`  | InfoNCE 文本对齐 | `lambda_nce = 0` |
| `no_kl_anneal` | CVAE β 退火（保持固定值）| `--no-kl-anneal` CLI |

## 文件清单（结构同 module/）

| 文件 | 作用 |
|---|---|
| `variants.py`     | 6 个变体的 single source of truth |
| `make_config.py`  | base config 按变体 patch 到 `configs/_ablation_loss/<variant>/` |
| `train_a.sh`      | 仅 A 训练循环（不做 B fine-tune）|
| `eval_all.sh`     | A test 上跑 4 项 eval |
| `aggregate.py`    | 输出 `runs/loss/_aggregate/table_loss.{csv,md}` |

## 在 8×H100 上的执行流程

```bash
cd /workspace/CAP

# 1) 训 6 个 A 变体 (~30 GPU·h)
bash eval/ablation/loss/train_a.sh

# 2) 跑 eval (~6 × 8 min ≈ 1 h)
bash eval/ablation/loss/eval_all.sh

# 3) 聚合
python eval/ablation/loss/aggregate.py
```

## 总成本

| 阶段 | 单变体 | × 6 | 总 |
|---|---|---|---|
| Train A | ~5 h | 30 h | **30 h** |
| Eval | ~10 min | 1 h | **1 h** |
| **TOTAL** | | | **~31 GPU·h** |

## 跟 Tier 1 (module/) 的关系

Tier 1 `no_algebraic` = Tier 2 `no_L_clos + no_L_inv + no_L_eq + no_L_comm` 同时关。所以两表互补：
- Tier 1 Tab 6：粗粒度（一组 loss 一起关）
- Tier 2 Tab S1（附录）：细粒度（每个 loss 单独关）

如果 Tier 2 显示某一项单消的影响很小（其他 lambda 不变），说明它跟同组其他 loss **冗余**——这是论文 Discussion 段的有用素材。

## 加变体怎么扩

编辑 `variants.py` 加 dict entry。其他文件不动。
