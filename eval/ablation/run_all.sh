#!/usr/bin/env bash
# run_all.sh — Run *every* ablation: K-sweep + Tier-1 module + Tier-2 loss.
#
# Sequential on 8 GPUs.  Total wall-clock estimate on 8×H100:
#
#   K-sweep    (4 × ~5 h A train + eval)              = ~21 h
#   Module A   (6 × ~5 h A train)                     = ~30 h
#   Module B   (6 × ~30 min B fine-tune)              =  ~3 h
#   Module evals                                      =  ~1 h
#   Loss A     (6 × ~5 h A train)                     = ~30 h
#   Loss evals                                        =  ~1 h
#                                                       ─────
#                                                       ~86 h ≈ 3.6 days
#
# Resumable: each individual variant has an "idempotency check" — if its
# ckpt already exists, the loop skips it.  ctrl-C and restart freely.
#
# Run from /workspace/CAP/:
#     nohup bash eval/ablation/run_all.sh > ablation_master.log 2>&1 &
#
# Skip B fine-tune for Tier 1 (saves ~4 GPU·h, A-only ablation is standard
# for NeurIPS):
#     SKIP_B=1 bash eval/ablation/run_all.sh

set -euo pipefail

if [[ ! -d configs ]] || [[ ! -d train ]] || [[ ! -d eval ]]; then
    echo "ERROR: run from CAP root.  cwd=$(pwd)"; exit 1
fi

ts() { date +"%Y-%m-%d %H:%M:%S"; }
section() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  $(ts)  $1"
    echo "═══════════════════════════════════════════════════════════════════"
}

# ──────────────────────────────────────────────────────────────────────
# Phase 1: K-sweep (Theorem 1 verification)
# ──────────────────────────────────────────────────────────────────────
section "Phase 1: K-sweep training"
bash eval/ablation/ksweep/train_sweep.sh

section "Phase 1: K-sweep eval + plot"
bash eval/ablation/ksweep/eval_sweep.sh
python eval/ablation/ksweep/plot_theorem1.py \
    --summary runs/ablation/ksweep/_eval/summary.json \
    --output  runs/ablation/ksweep/_eval/theorem1.pdf || echo "(plot failed, see eval output)"

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Module-level (Tier 1) — A then B then eval
# ──────────────────────────────────────────────────────────────────────
section "Phase 2.1: Module ablations — A train"
bash eval/ablation/module/train_a.sh

if [[ "${SKIP_B:-0}" == "0" ]]; then
    section "Phase 2.2: Module ablations — B fine-tune"
    bash eval/ablation/module/finetune_b.sh
else
    section "Phase 2.2: SKIPPED (SKIP_B=1)"
fi

section "Phase 2.3: Module ablations — eval"
bash eval/ablation/module/eval_all.sh   # auto-skips missing B ckpts

section "Phase 2.4: Module ablations — aggregate"
if [[ "${SKIP_B:-0}" == "0" ]]; then
    python eval/ablation/module/aggregate.py
else
    python eval/ablation/module/aggregate.py --datasets a
fi

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Loss-level (Tier 2) — A only
# ──────────────────────────────────────────────────────────────────────
section "Phase 3.1: Loss ablations — A train"
bash eval/ablation/loss/train_a.sh

section "Phase 3.2: Loss ablations — eval"
bash eval/ablation/loss/eval_all.sh

section "Phase 3.3: Loss ablations — aggregate"
python eval/ablation/loss/aggregate.py

# ──────────────────────────────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────────────────────────────
section "ALL ABLATIONS COMPLETE"
echo ""
echo "  Outputs:"
echo "    K-sweep        : runs/ablation/ksweep/_eval/{summary.json, theorem1.pdf}"
echo "    Module (Tab 6) : runs/ablation/module/_aggregate/table6.{csv,md}"
echo "    Loss   (Tab S1): runs/ablation/loss/_aggregate/table_loss.{csv,md}"
echo ""
echo "  $(ts)  done."
