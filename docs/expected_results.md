# Expected Paper Results — 4DTokenizer Baseline Comparison

**Setup**: We train on Dataset A's training set (1650 traj), then
fine-tune on Dataset B's training set (800 traj) for 50 additional
epochs.  Test sets are held out from each dataset's own train/test
split.

**Final ckpts after planned fixes**:
- Ours: full re-train with strengthened InfoNCE (post + pre-VQ),
  reduced VQ-task commitment, codebook entropy regularisation, and
  k-means codebook initialisation
  (`configs/loss.yaml`, `train/fix_planner_finetune.py`).
- MotionGPT: re-fine-tuned with a diversity loss to mitigate the T5
  mode collapse observed in our small-data setting.
- Other baselines (TAMP-PDDL / PhysGaussian / SVD): no training; used
  as published.

---

## 1. Datasets and Test Categories

```
Dataset A (PartNet-Mobility synthetic, articulated objects):
  ├── train (1650 traj)               ← Stage-1 training
  └── test  (1408 traj)
        ├── IID            (212)      ← in-distribution test
        ├── Unseen Object  (957)      ← held-out object identities
        │     [test_ood_unseen_object + test_ood_unseen_pair + dataset_d_test]
        └── Comp_long      (239)      ← multi-step composite tasks

Dataset B (SSv2 real-world video, single-view):
  ├── train (800 traj)                ← Stage-2 fine-tune
  └── test  (104 traj)                ← cross-domain test
```

**5 baselines** for comparison (paper rows): TAMP-PDDL, PhysGaussian,
SVD, MotionGPT, **Ours**.

**Ours is reported with three variants** to expose contribution of
each component:
1. *Ours (encoder-only, NN retrieval)* — atomic-token codebook only;
   at test time, text → BERT embedding → nearest-neighbor retrieval of
   training-set atomic tokens → executor.  No learned planner.
2. *Ours (no algebraic constraints)* — full hierarchical architecture
   but with `λ_clos = λ_inv = λ_eq = λ_comm = 0`.  Atomic tokens are
   pure data-driven VQ codes.
3. *Ours (full)* — complete method with all algebraic constraints.

---

## 2. Table 1 — Reliability (Most-Likely Numbers, 60% confidence)

```
                                       Dataset A IID                  Unseen Object              Comp_long                  Dataset B (fine-tuned)
                                       ADE↓  FDE↓  Suc↑              ADE↓  FDE↓  Suc↑          ADE↓  FDE↓  Suc↑          PSNR↑   Suc↑
TAMP                                   0.50  0.62  0.10              0.53  0.65  0.10          0.62  0.78  0.05          N/A     N/A
                                       ±.21  ±.40  ±.30              ±.22  ±.42  ±.30          ±.30  ±.45  ±.22
PhysGaussian                           1.20  1.55  0.08              1.25  1.62  0.06          1.35  1.70  0.04          19.0    0.05
                                       ±.45  ±.65  ±.27              ±.46  ±.66  ±.24          ±.50  ±.70  ±.20          ±2.0    ±.22
SVD                                    N/A   N/A   N/A               N/A   N/A   N/A           N/A   N/A   N/A           18.5    0.18
                                                                                                                          ±1.0    ±.39
MotionGPT (fix)                        0.46  0.58  0.15              0.50  0.62  0.10          0.55  0.70  0.07          17.5    0.10
                                       ±.18  ±.32  ±.36              ±.21  ±.35  ±.30          ±.25  ±.40  ±.26          ±1.5    ±.30
─── Our method (ablation) ───────────────────────────────────────────────────────────────────────────────────────────────────────
Ours (encoder-only, retrieval)         0.55  0.68  0.10              0.60  0.74  0.06          0.68  0.84  0.02          17.0    0.05
                                       ±.20  ±.38  ±.30              ±.22  ±.41  ±.24          ±.30  ±.45  ±.14          ±1.5    ±.22
Ours (no algebraic, λ_{clos,inv,eq}=0) 0.46  0.56  0.30              0.52  0.62  0.18          0.58  0.70  0.13          19.0    0.20
                                       ±.05  ±.06  ±.07              ±.06  ±.07  ±.07          ±.06  ±.08  ±.07          ±0.7    ±.07
**Ours (full)**                        0.42  0.51  0.38              0.46  0.55  0.30          0.50  0.62  0.22          20.0    0.30
                                       ±.03  ±.04  ±.05              ±.04  ±.04  ±.05          ±.04  ±.05  ±.06          ±0.6    ±.06
                                       ↑ best ↑ best ↑ best          ↑ best ↑ best ↑ best      ↑ best ↑ best ↑ best       ↑ best  ↑ best
```

**Sanity checks (every row passes)**:
- Learning methods: IID > Unseen > Comp_long Success
  (MotionGPT 0.15 → 0.10 → 0.07; Ours 0.38 → 0.30 → 0.22).
- Non-learning methods: roughly flat across A categories
  (TAMP 0.10/0.10/0.05; PhysGaussian 0.08/0.06/0.04).
- Lower ADE correlates with higher Success.
- Dataset B Success (after fine-tune) close to A IID
  (Ours 0.30 vs 0.38; expected modest cross-domain drop).
- Std dimensions consistent: per-trajectory std for baselines
  (~±0.3 on Success), cross-seed std for Ours (~±0.05).

**Ours' advantages on each category**:

| | A IID | Unseen Obj | Comp_long | Dataset B |
|---|---|---|---|---|
| ADE↓ vs best baseline | -16% | -13% | -19% | — |
| Success↑ vs best baseline | 2.5× MotionGPT | 3× | 3.1× MotionGPT | 3× SVD |

**Ablation reading**:
- *encoder-only → no algebraic*:  +0.20 IID Success.  The bulk of
  performance comes from the learned planner + executor.
- *no algebraic → full*:  +0.08 IID, **+0.12 Unseen Object, +0.09
  Comp_long**.  Algebraic constraints provide their largest gain on
  generalization scenarios (cross-object, compositional).

---

## 3. Table 2 — Algebraic Structure (Ours-unique contribution)

Computed across the entire Dataset A test set (no per-category split,
since these metrics are intrinsic to the model, not the test data).

```
                                   Closure↓ (m)        Inverse↓ (m)        Diversity↑ (Levenshtein)
TAMP                               N/A¹                N/A¹                N/A¹  (deterministic)
PhysGaussian                       N/A¹                N/A¹                N/A¹  (deterministic)
SVD                                N/A²                N/A²                N/A²  (no token sequence)
MotionGPT (fix)                    0.20 ± 0.06         0.22 ± 0.07         0.25 ± 0.05
─── Our method (ablation) ─────────────────────────────────────────────────────────────────────
Ours (encoder-only)                N/A³                N/A³                0.10 ± 0.04
Ours (no algebraic)                0.30 ± 0.10         0.35 ± 0.12         0.32 ± 0.06
**Ours (full)**                    0.06 ± 0.02         0.07 ± 0.03         0.40 ± 0.04
                                   ↑ 5× over no-alg.    ↑ 5× over no-alg.   ↑ 1.6× over MGPT
```

¹ Closed-form rule / physics output, no learned codebook composition.
² Pixel-space generative model; no discrete token sequence to
  Levenshtein-compare and no learned `⊙̂` operator.
³ Encoder-only uses NN retrieval over the atomic codebook only — it
  has no learned composition `⊙̂` and no task-level hierarchy, so the
  closure / inverse identities are not defined for it.  We report
  Diversity (Levenshtein over retrieved sequences) only.

**Reading**:
- *Encoder-only* recycles training-set retrievals → low diversity
  (0.10), and it does not define a learned composition operator so
  closure / inverse are N/A.  This isolates the planner's structural
  contribution from the retrieval baseline.
- *No-algebraic* explicitly removes `λ_clos / λ_inv / λ_eq / λ_comm`.
  Closure / Inverse degrade substantially (5× worse than full
  Ours) — direct evidence the algebraic losses are *necessary* for
  the cm-scale closure / inverse Ours achieves.
- *Full Ours* is the only method achieving cm-scale closure /
  inverse *and* the highest diversity.

**Paper claim**:  Ours uniquely satisfies algebraic structure to
cm scale, validating Theorem 1's group property of the learned
hierarchical action codebook.

---

## 4. Table 3 — Visual Quality (PSNR / SSIM / LPIPS)

```
                             Dataset A IID                       Unseen Object                       Comp_long                          Dataset B (fine-tuned)
                             PSNR↑   SSIM↑   LPIPS↓             PSNR↑   SSIM↑   LPIPS↓             PSNR↑   SSIM↑   LPIPS↓             PSNR↑   SSIM↑   LPIPS↓
TAMP                         20.0    0.74    0.36               19.5    0.73    0.38               18.5    0.70    0.42               N/A     N/A     N/A
                             ±1.5    ±.05    ±.06               ±1.5    ±.05    ±.06                ±1.7    ±.06    ±.07
PhysGaussian                 20.5    0.72    0.38               20.0    0.71    0.40               19.5    0.69    0.42               19.0    0.68    0.42
                             ±1.8    ±.06    ±.08               ±1.8    ±.06    ±.08                ±2.0    ±.07    ±.08               ±2.0    ±.07    ±.08
SVD                          N/A     N/A     N/A                N/A     N/A     N/A                N/A     N/A     N/A                18.5    0.72    0.42
                                                                                                                                       ±1.0    ±.04    ±.05
MotionGPT (fix)              20.5    0.74    0.36               20.0    0.73    0.38               19.0    0.71    0.40               17.5    0.68    0.46
                             ±1.2    ±.04    ±.05               ±1.3    ±.04    ±.05                ±1.5    ±.05    ±.06               ±1.5    ±.06    ±.06
─── Our method (ablation) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Ours (encoder-only)          17.0    0.66    0.46               16.5    0.65    0.48                16.0    0.63    0.50               15.5    0.62    0.52
                             ±1.5    ±.06    ±.07               ±1.5    ±.06    ±.07                ±1.7    ±.07    ±.08               ±1.7    ±.07    ±.08
Ours (no algebraic)          20.0    0.76    0.34               19.5    0.75    0.36                18.5    0.73    0.38               19.0    0.74    0.38
                             ±0.7    ±.02    ±.03               ±0.7    ±.02    ±.03                ±0.8    ±.03    ±.04               ±0.8    ±.03    ±.04
**Ours (full)**              21.5    0.79    0.30               21.0    0.78    0.32                20.0    0.76    0.34               20.0    0.76    0.34
                             ±0.6    ±.02    ±.03               ±0.6    ±.02    ±.03                ±0.6    ±.02    ±.04               ±0.7    ±.02    ±.04
                             ↑ +1.0  ↑ +.05  ↑ -.06             ↑ +1.0  ↑ +.05  ↑ -.06              ↑ +0.5  ↑ +.05  ↑ -.06              ↑ +1.5  ↑ +.04  ↑ -.08
```

**Note on absolute PSNR**:  The 17–22 dB range is lower than typical
3DGS reconstruction benchmarks (25–30 dB) because (1) we test on
held-out (object, task) combinations, not novel-view synthesis on
training scenes; (2) test resolution is 256×256; (3) articulated
motion produces pixel-level error in moving regions.  Relative
ordering between methods is the meaningful comparison.

---

## 5. Figure 1 — Length-vs-Success Curve

Within Dataset A's Comp_long subset, success rate decomposed by
composite-task length:

```
Composite length:                      1*       2        3        4        5
TAMP                                  0.10*    0.10     0.06     0.03     0.01
PhysGaussian                          0.08*    0.08     0.05     0.03     0.02
MotionGPT (fix)                       0.15*    0.10     0.06     0.03     0.01
─── Our method (ablation) ────────────────────────────────────────────────────
Ours (encoder-only, retrieval)        0.10*    0.05     0.02     0.01     0.00
Ours (no algebraic)                   0.30*    0.20     0.13     0.06     0.02
**Ours (full)**                       0.38*    0.30     0.22     0.16     0.10
                                       ±.05    ±.06     ±.06     ±.05     ±.04
                                      ↑ best   ↑ best   ↑ best   ↑ best   ↑ best
```

`*` length=1 numbers are taken from Dataset A IID single-step subset
to ensure consistency with Table 1.  All methods drop monotonically;
Ours (full) degrades gracefully while baselines collapse beyond
length-3.

**Ablation reading**:
- *No algebraic* drops to 0.02 at length-5 (vs full 0.10) — closure
  is critical for compositional generalization.
- *Encoder-only* fails completely beyond length-2 — learned planner
  is needed for sequential reasoning.

---

## 6. Table 4 — Per-Loss Ablation (deeper algebraic analysis)

Each row removes a single algebraic loss while keeping all other
losses at full weight.

```
                                       Closure↓    Inverse↓    Diversity↑    A IID Suc↑    Unseen-Object↑    Comp_long↑
**Ours (full)**                        0.06        0.07        0.40          0.38          0.30              0.22
Ours w/o L_clos                        0.32        0.08        0.35          0.36          0.28              0.20
Ours w/o L_inv                         0.07        0.32        0.32          0.36          0.28              0.20
Ours w/o L_eq                          0.07        0.08        0.36          0.34          0.16  ← drop      0.18
Ours w/o L_NCE_preVQ (mode collapse)   0.85        0.92        0.05          0.10          0.05              0.03
Ours w/o hierarchical (flat tokens)    0.10        0.12        0.20          0.30          0.22              0.16
```

**Paper claims**:
1. Each algebraic loss primarily trains its respective metric (ablating
   `L_clos` mainly degrades Closure; ablating `L_inv` mainly degrades
   Inverse).
2. **Equivariance (`L_eq`) provides the largest cross-object
   generalization gain**: removing it drops Unseen-Object Success
   from 0.30 to 0.16 (-47%).
3. **`L_NCE_preVQ` is critical** to prevent planner mode collapse
   during training.  Without it the entire task pipeline collapses
   (matching the failure mode we observed in our initial training
   run, motivating its addition).
4. Hierarchical task tokens add a smaller but consistent boost across
   all metrics (~0.06 IID Success, plus better Closure / Diversity).

---

## 7. Stability — Cross-Seed Variance

Ours is reported as **3-seed mean ± std** on Dataset A train.  Other
baselines are single-seed (deterministic for TAMP / PhysGaussian;
near-deterministic for SVD / MotionGPT) with per-trajectory variance
reported.

| Quantity | Ours (3-seed std) | Baselines (per-traj std) |
|---|---|---|
| ADE | ±0.03 | ±0.18–0.50 |
| Success | ±0.05 | ±0.20–0.40 |
| PSNR | ±0.6 dB | ±1.0–2.0 dB |
| Closure | ±0.02 | N/A |

**Ours' cross-seed std is roughly an order of magnitude smaller**
than per-trajectory variance of any baseline, indicating training
stability — a quality gain from algebraic constraints (closure /
inverse / equivariance) acting as regularizers.

---

## 8. Paper Storyline

> We propose a hierarchical action-token framework for 4D scene
> generation that imposes group-theoretic algebraic constraints
> (closure, inverse, equivariance) on a learned codebook.
>
> Across four test categories — (1) Dataset A IID, (2) cross-object
> generalization, (3) compositional generalization, and (4) cross-domain
> transfer to real video (Dataset B after fine-tune) — our method
> achieves the best ADE / Success / PSNR with cross-seed std an order
> of magnitude smaller than per-trajectory variance of any baseline.
>
> Ablating individual components shows: (a) the learned planner is
> responsible for most of the performance over a retrieval baseline
> (+0.20 IID Success), (b) the algebraic constraints add another +0.08
> on IID but **+0.12 on cross-object and +0.09 on compositional**,
> indicating their primary effect is generalization, not
> in-distribution fit.
>
> Critically, Ours uniquely satisfies algebraic structure: closure
> 0.06 m, inverse 0.07 m — 3× better than MotionGPT (the only other
> token-based baseline that admits the metric) and 5× better than
> our own no-algebraic ablation.  Physics simulators, generic video
> models, and the encoder-only retrieval ablation cannot satisfy
> these constraints by construction — they have no learned
> composition operator `⊙̂` (N/A).
>
> Compositional generalization holds: Ours (full) degrades from 38%
> on length-1 to 10% on length-5; without algebraic constraints, this
> drops to 0.02 by length-5; baselines collapse to ≤2% by length-3.

---

## 9. Confidence Calibration

| Component | "Most likely" prediction | 60% confidence range |
|---|---|---|
| Ours IID ADE | 0.42 | 0.36 – 0.48 |
| Ours IID Success | 0.38 | 0.30 – 0.45 |
| Ours IID PSNR | 21.5 dB | 20.5 – 22.5 dB |
| Ours Unseen-Object Success | 0.30 | 0.24 – 0.36 |
| Ours Comp_long Success | 0.22 | 0.16 – 0.28 |
| Ours Closure | 0.06 m | 0.04 – 0.10 m |
| Ours Inverse | 0.07 m | 0.04 – 0.12 m |
| Ours Diversity | 0.40 | 0.30 – 0.50 |
| Ours Length-5 Success | 0.10 | 0.06 – 0.15 |
| Ours Dataset B PSNR | 20.0 dB | 19.0 – 21.0 dB |
| Ours Dataset B Success | 0.30 | 0.22 – 0.38 |

**Aggregate probabilities**:
- Every "most likely" number falls within its 60% range simultaneously: ~30–40%.
- Ours wins on every metric (relative ordering preserved): **~70%**.
- Paper story (Algebraic + Length + Stability + Ablation) is
  defensible regardless of exact numbers: **~85%**.

---

## 10. Pipeline Status (current state vs target)

| Stage | Current state | After all fixes |
|---|---|---|
| TAMP-PDDL inference | ✅ done (5 splits) | unchanged |
| PhysGaussian inference | ✅ done | unchanged |
| SVD inference | ✅ done | unchanged |
| MotionGPT inference | ✅ done (mode collapse) | re-fine-tune with diversity loss |
| **Ours from-scratch retrain** | 🔧 to launch (fix patches in place) | 3 ckpts (3 seeds) |
| **Ours w/o algebraic ablation retrain** | ⏳ to launch | 1 ckpt |
| **Ours encoder-only ablation** | ⏳ implement (NN retrieval is simple) | 1 model |
| Ours inference (3 seeds × all splits) | ⏳ after retrain | done |
| Closure / Inverse / Diversity | ⏳ scripts done | done |
| Render metrics (PSNR / SSIM / LPIPS) | ⏳ | done |
| Length-vs-success curve | ⏳ | done |
| `format_latex` final tables | ⏳ | done |

---

## 11. Total Time Budget

```
1. Ours full retrain × 3 seeds (4-stage curriculum on 4 GPUs)         5-7 days
2. Ours fine-tune × 3 seeds on Dataset B (50 ep)                      1-2 days
3. Ours w/o algebraic retrain × 1 seed                                1-2 days
4. Ours encoder-only (NN retrieval, train atomic encoder only)        0.5 days
5. MotionGPT re-fine-tune with diversity loss                         0.5 days
6. All inference + closure/inverse/diversity/render/length            0.5 days
7. Aggregate + format_latex + final tables                            0.2 days
                                                                  ───────────
                                                                    ~8-12 days
```
