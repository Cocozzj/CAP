#!/usr/bin/env bash
# run_all.sh — Run all ablations on Dataset-A: K-sweep + module Tier-1 + loss Tier-2.
#
# A-only by design.  Ablations target architectural / loss design which is
# dataset-agnostic, so we focus on the synthetic setting (Dataset-A) where
# closure / inverse / commutator have exact GT.  The pre-trained main
# checkpoints at runs/main_a/seed_{0,1,2}/ provide the baseline row.
#
# Sequential on 8 GPUs.  Wall-clock estimate on 8×H100 with 80 ep ablations:
#
#   K-sweep    (4 K × ~2.7 h A train @ 80ep + eval)    = ~12 h   (K=2048 skipped)
#   Module A   (6 var × ~2.7 h A train @ 80ep)         = ~16 h
#   Module evals                                       =  ~1 h
#   Loss A     (6 var × ~2.7 h A train @ 80ep)         = ~16 h
#   Loss evals                                         =  ~1 h
#                                                        ─────
#                                                        ~46 h ≈ 2 days
#
# Resumable: each individual variant has an "idempotency check" — if its
# ckpt already exists, the loop skips it.  ctrl-C and restart freely.
#
# Run from /workspace/CAP/:
#     nohup bash eval/ablation/run_all.sh > runs/ablation_master.log 2>&1 &

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
section "Phase 1.1: K-sweep training"
bash eval/ablation/ksweep/train_sweep.sh

section "Phase 1.2: K-sweep eval + plot"
bash eval/ablation/ksweep/eval_sweep.sh
python eval/ablation/ksweep/plot_theorem1.py \
    --summary runs/ksweep/_eval/summary.json \
    --output  runs/ksweep/_eval/theorem1.pdf || echo "(plot failed, see eval output)"

# ──────────────────────────────────────────────────────────────────────
# Phase 2: Module-level ablations (Tier 1) — A only
# ──────────────────────────────────────────────────────────────────────
section "Phase 2.1: Module ablations — A train"
bash eval/ablation/module/train_a.sh

section "Phase 2.2: Module ablations — eval"
bash eval/ablation/module/eval_all.sh

section "Phase 2.3: Module ablations — aggregate"
python eval/ablation/module/aggregate.py

# ──────────────────────────────────────────────────────────────────────
# Phase 3: Loss-term ablations (Tier 2) — A only
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
echo "    K-sweep        : runs/ksweep/_eval/{summary.json, theorem1.pdf}"
echo "    Module (Tab 6) : runs/module/_aggregate/table6.{csv,md}"
echo "    Loss   (Tab S1): runs/loss/_aggregate/table_loss.{csv,md}"
echo ""
echo "  $(ts)  done."
