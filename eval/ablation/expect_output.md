# Expected ablation outputs (post-fix, retrained from scratch)

> Predicted results assuming: codebook-collapse bug fixed, all other pipeline
> issues resolved, every variant trained from scratch under matched curriculum.
>
> All numbers below are **conservative most-likely estimates** based on:
> 1. The single grounded measurement (`main_a` closure_gap = 0.014 m from
>    K-sweep on physics-off eval)
> 2. Typical NeurIPS ablation effect-size literature (median 1.3-2.2×)
> 3. Theoretical role of each ablated component in the loss suite
>
> **Expect ±25% noise per cell**.  Patterns matter more than exact numbers.

## Tab 6 — Module ablation

```
| variant       | dataset | Δ_clos (m)        | Δ_inv (m)         | ADE (m)         | FDE (m)         | Success rate    | Lev diversity   | Codebook used |
|---------------|---------|-------------------|-------------------|-----------------|-----------------|-----------------|-----------------|---------------|
| main (n=3)    | a       | 0.014 ± 0.003     | 0.013 ± 0.002     | 0.085 ± 0.010   | 0.170 ± 0.018   | 0.58 ± 0.04     | 0.38 ± 0.04     | 95%           |
| no_algebraic  | a       | 0.028 (2.0×)      | 0.026 (2.0×)      | 0.094 (1.1×)    | 0.187 (1.1×)    | 0.46 (-12 pp)   | 0.36            | 82%           |
| no_physics    | a       | 0.017 (1.2×)      | 0.015 (1.2×)      | 0.140 (1.6×)    | 0.282 (1.7×)    | 0.45 (-13 pp)   | 0.37            | 93%           |
| no_hier       | a       | 0.019 (1.4×)      | 0.018 (1.4×)      | 0.092 (1.1×)    | 0.184 (1.1×)    | 0.43 (-15 pp)   | 0.39            | 92%           |
| no_cvae       | a       | 0.017 (1.2×)      | 0.016 (1.2×)      | 0.087 (1.0×)    | 0.174 (1.0×)    | 0.52 (-6 pp)    | 0.06 (-84%)     | 88%           |
```

### Per-row most-likely story
- `no_algebraic`: closure / inverse **2× up**, success **-12 pp**
- `no_physics`: ADE/FDE **1.6-1.7×** (trajectory drift), success **-13 pp**
- `no_hier`: success **-15 pp** (composite tasks broken), other metrics minor
- `no_cvae`: diversity **collapses 84%**, other metrics roughly unchanged

## Tab S1 — Loss-term ablation (theorem-aligned)

```
| variant      | Δ_clos          | Δ_inv           | Δ_eq            | Δ_hier         | NCE-MI         | Success         |
|--------------|-----------------|-----------------|-----------------|----------------|----------------|-----------------|
| main (n=3)   | 0.014 ± 0.003   | 0.013 ± 0.002   | 0.022 ± 0.004   | 0.019 ± 0.003  | 0.78 ± 0.05    | 0.58 ± 0.04     |
| no_L_clos    | 0.024 (1.7×)    | 0.016 (1.2×)    | 0.024 (1.1×)    | 0.020 (1.1×)   | 0.77           | 0.53 (-5 pp)    |
| no_L_inv     | 0.018 (1.3×)    | 0.022 (1.7×)    | 0.024 (1.1×)    | 0.020 (1.1×)   | 0.77           | 0.53 (-5 pp)    |
| no_L_eq      | 0.017 (1.2×)    | 0.016 (1.2×)    | 0.040 (1.8×)    | 0.022 (1.2×)   | 0.76           | 0.51 (-7 pp)    |
| no_L_hier    | 0.017 (1.2×)    | 0.016 (1.2×)    | 0.024 (1.1×)    | 0.034 (1.8×)   | 0.75           | 0.49 (-9 pp)    |
| no_L_nce     | 0.016 (1.1×)    | 0.015 (1.1×)    | 0.023 (1.0×)    | 0.020 (1.1×)   | 0.50 (-36%)    | 0.51 (-7 pp)    |
```

### Pattern
- Diagonal cells: **1.7-1.8×** (each loss has visible but moderate impact on its metric)
- Off-diagonal cells: **1.0-1.2×** (mostly within-noise; some compensation)
- `no_L_nce` mainly hits NCE-MI (-36%); algebraic gaps unchanged because InfoNCE
  doesn't impose token structure
- Success drops **5-9 pp** uniformly — single-loss removal is recoverable

## Tab S3 — Codebook utilisation

```
| variant      | K   | unique | frac used | norm mean ± std    | health      |
|--------------|-----|--------|-----------|--------------------|-------------|
| main (n=3)   | 512 | 488    | 95.3%     | 1.812 ± 0.054      | ✅ healthy  |
| no_algebraic | 512 | 421    | 82.2%     | 1.692 ± 0.182      | 🟢 mostly   |
| no_physics   | 512 | 475    | 92.8%     | 1.795 ± 0.078      | ✅ healthy  |
| no_hier      | 512 | 471    | 92.0%     | 1.788 ± 0.082      | ✅ healthy  |
| no_cvae      | 512 | 451    | 88.1%     | 1.748 ± 0.115      | 🟢 mostly   |
| no_L_clos    | 512 | 442    | 86.3%     | 1.728 ± 0.132      | 🟢 mostly   |
| no_L_inv     | 512 | 458    | 89.5%     | 1.755 ± 0.108      | 🟢 mostly   |
| no_L_eq      | 512 | 466    | 91.0%     | 1.772 ± 0.092      | 🟢 mostly   |
| no_L_hier    | 512 | 478    | 93.4%     | 1.798 ± 0.064      | ✅ healthy  |
| no_L_nce     | 512 | 487    | 95.1%     | 1.808 ± 0.048      | ✅ healthy  |
```

### Key pattern
- Variants that remove **algebraic constraints** (`no_algebraic`, `no_L_clos`,
  `no_L_inv`) drop to **82-89%** because group structure was the main pressure
  spreading codes
- Variants that preserve algebra (`no_physics`, `no_hier`, `no_L_nce`) stay
  near `main` (92-95%)
- `no_cvae` drops moderately (88%) — deterministic Planner reduces commitment
  loss diversity but doesn't fully break codebook spread

## Tab S5 — Per-stage val_loss

```
| variant       | RIGID  | PLANNER | PHYSICS | FULL          | Δ vs main FULL  |
|---------------|--------|---------|---------|---------------|-----------------|
| main (n=3)    | 6.05   | 5.65    | 5.78    | 5.85 ± 0.20   | —               |
| no_algebraic  | 6.45   | 6.20    | 6.85    | 7.20          | +23%            |
| no_physics    | 6.20   | 5.62    | 5.65    | 0.92          | -84% (artifact) |
| no_hier       | 6.30   | 5.85    | 6.00    | 6.50          | +11%            |
| no_cvae       | 6.30   | 6.50    | 9.20    | 10.80         | +85%            |
| no_L_clos     | 6.20   | 5.80    | 7.10    | 7.40          | +27%            |
| no_L_inv      | 6.20   | 5.80    | 7.05    | 7.30          | +25%            |
| no_L_eq       | 6.30   | 6.05    | 8.50    | 9.20          | +57%            |
| no_L_hier     | 6.30   | 6.10    | 8.80    | 9.50          | +62%            |
| no_L_nce      | 6.20   | 5.95    | 8.60    | 9.30          | +59%            |
```

### Stage-wise diverge story
- All variants tightly grouped in RIGID (within ~5% of main)
- Divergence widens at PHYSICS stage (when full curriculum activates)
- FULL stage shows clearest separation
- `no_physics` artifact: physics_total=0 always pulls total down
- After codebook trains properly, **divergence appears earlier** than in the
  frozen-codebook setting (which only diverged at FULL stage)

### 4-tier groups (post-fix)
```
Tier 0  (artifact):     no_physics                  ≈ 0.9
Tier 1  (close to main): main, no_hier              5.85 - 6.5
Tier 2  (moderate):     no_algebraic, no_L_clos,    7.2 - 7.4
                        no_L_inv
Tier 3  (severe):       no_L_eq, no_L_hier,         9.2 - 10.8
                        no_L_nce, no_cvae
```

## Fig 4 — Val curves overlay

Curves diverge at end of RIGID stage and continue to spread.  Final-epoch
positions:

```
                  RIGID   PLANNER  PHYSICS    FULL
val_loss              ╲       ╲       ╲        ↓
   12 ─                                     no_cvae ━━━
      │                                     no_L_hier ╴╴
   10 ─                                     no_L_nce ──
      │                                     no_L_eq ┄┄
    8 ─                                     no_L_clos ─
      │                                     no_L_inv ╶╶
      │                                     no_algebraic ━
    6 ─━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ main ▬▬▬
      │                                     no_hier ━━━
    4 ─
    2 ─
    1 ─━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ no_physics
      └─────┬───────┬───────┬──────────────→
            25      40      55             95   epoch
```

## Cross-table consistency checks

If predictions hold, these should be self-consistent:

1. Variants with **closure_gap up >1.5×** in Tab 6 should also appear in
   **codebook frac_used < 90%** in Tab S3:
   - ✓ `no_algebraic`: closure 2.0× ↔ codebook 82%
2. Variants with **success drop > -10 pp** should also have **higher FULL val_loss**:
   - ✓ `no_algebraic`: -12 pp ↔ +23% val
   - ✓ `no_hier`: -15 pp ↔ +11% val (slight inconsistency, val less sensitive)
3. `no_cvae` diversity collapse should NOT hurt closure/inverse:
   - ✓ closure 1.2× (within-noise), diversity -84%

If the real run violates many of these consistency checks, **investigate before
writing the paper**.

## What conservative-realistic predictions look like vs idealised

| Aspect | Conservative (this file) | Idealised (don't expect) |
|---|---|---|
| Diagonal effect size | 1.7-1.8× | 5-15× |
| Off-diagonal | 1.0-1.2× | exactly 1.0× |
| Success drops | 5-15 pp | 30+ pp |
| Codebook range | 82-95% | 17-100% |
| Diversity collapse | -84% | -100% |
| Tier separation | gradual | crisp |

## Honest uncertainty

| Cell | Confidence | Why |
|---|---|---|
| `main` baselines (Tab 6 first row) | medium | anchored on K-sweep K=512 = 0.014 |
| `no_physics` Tab 6 ADE/FDE 1.6-1.7× | medium | structural effect well-defined |
| `no_cvae` diversity -84% | high | structural certainty (deterministic = no diversity) |
| Loss ablation diagonal magnitudes | medium-low | could be 1.4× or 2.2× |
| Val_loss specific numbers | low | retraining dynamics genuinely uncertain |
| Codebook utilisation per variant | low | depends on fix details |
| Per-task success breakdown | very low | task definitions matter |

## Bottom line

This represents **the typical NeurIPS-grade ablation table you'd expect from a
correctly-implemented model with one specific component removed at a time**.
Effect sizes are visible (1.3-2.0×) but not dramatic; some cells are statistical
noise; results require interpretation rather than just reading numbers.

Most cells should land within ±25% of these predictions when you re-run.
Cells that deviate by 2× or more become **paper Discussion material**, not bugs.
