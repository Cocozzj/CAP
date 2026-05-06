#!/usr/bin/env bash
# eval_all.sh — A-only eval suite for Tier-2 loss ablations.
#
# Mirror of module/eval_all.sh but skips the B half (Tier 2 doesn't
# fine-tune on B).  Each variant's eval lands in
# runs/loss/<variant>/seed_<S>/eval_a/<eval>/.

set -euo pipefail

VARIANTS="${VARIANTS:-no_L_clos no_L_inv no_L_comm no_L_hier no_L_nce no_kl_anneal}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-runs/loss}"

A_MANIFEST="${A_MANIFEST:-dataset/dataset_a/manifest.json}"
A_DATA="${A_DATA:-dataset/dataset_a/data}"
A_SPLIT="${A_SPLIT:-test_iid}"
TASKS="${TASKS:-open_drawer close_drawer pull_handle push_button rotate_knob}"

N_BATCHES="${N_BATCHES:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ ! -d eval ]]; then
    echo "ERROR: run from CAP root.  cwd=$(pwd)"; exit 1
fi

run_eval_suite() {
    local CKPT="$1"; local OUT="$2"
    if [[ ! -f "$CKPT" ]]; then
        echo "    skip: $CKPT (missing)"; return
    fi
    mkdir -p "$OUT"

    python -m eval.algebraic_gaps \
        --ckpt "$CKPT" --output-dir "$OUT/algebraic_gaps" \
        --dataset a --manifest "$A_MANIFEST" --data-dir "$A_DATA" --split "$A_SPLIT" \
        --n-batches "$N_BATCHES" --batch-size "$BATCH_SIZE" \
        --enable-physics  || echo "      [algebraic_gaps] failed"

    python -m eval.trajectory_metrics \
        --ckpt "$CKPT" --output-dir "$OUT/trajectory_metrics" \
        --dataset a --manifest "$A_MANIFEST" --data-dir "$A_DATA" --split "$A_SPLIT" \
        --n-batches "$N_BATCHES" --batch-size "$BATCH_SIZE" \
        --enable-physics  || echo "      [trajectory_metrics] failed"

    python -m eval.success_rate \
        --ckpt "$CKPT" --output-dir "$OUT/success_rate" \
        --dataset a --manifest "$A_MANIFEST" --data-dir "$A_DATA" --split "$A_SPLIT" \
        --tasks $TASKS --zero-shot --enable-physics  \
        || echo "      [success_rate] failed"

    python -m eval.diversity \
        --ckpt "$CKPT" --output-dir "$OUT/diversity" \
        --dataset a --manifest "$A_MANIFEST" --data-dir "$A_DATA" --split "$A_SPLIT" \
        --num-samples 16  || echo "      [diversity] failed"
}

for V in $VARIANTS; do
    A_CKPT="${RUN_ROOT}/${V}/seed_${SEED}/ckpt/main_exp_final.pt"
    echo ""
    echo "═══ eval ${V}  seed=${SEED} ═══"
    run_eval_suite "$A_CKPT" "${RUN_ROOT}/${V}/seed_${SEED}/eval_a"
done

# Main reference — A on all 3 seeds for mean±std baseline.
MAIN_SEEDS="${MAIN_SEEDS:-0 1 2}"
MAIN_OUT="${MAIN_OUT:-runs/loss/_main}"
for MS in $MAIN_SEEDS; do
    A_CK="runs/main_a/seed_${MS}/ckpt/main_exp_final.pt"
    if [[ ! -f "$A_CK" ]]; then
        echo "  main seed=${MS} A ckpt not found, skipping"
        continue
    fi
    echo ""
    echo "═══ eval main  seed=${MS} ═══"
    run_eval_suite "$A_CK" "${MAIN_OUT}/seed_${MS}/eval_a"
done

echo ""
echo "═══ All Tier 2 evals done.  Next: python eval/ablation/loss/aggregate.py ═══"
