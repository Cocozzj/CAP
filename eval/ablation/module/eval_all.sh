#!/usr/bin/env bash
# eval_all.sh — Run the eval suite on every module-ablation variant.
#
# Per variant × {A ckpt, B ckpt} runs:
#   - algebraic_gaps     (closure / inverse / commutator)
#   - trajectory_metrics (ADE / FDE / MPJPE)
#   - success_rate       (--zero-shot, on held-out split)
#   - diversity          (Levenshtein + codebook usage)
#
# Output: runs/module/<variant>/seed_<S>/{eval_a,eval_b}/<eval>/results.json
# These get aggregated by aggregate.py into Tab 6.
#
# Run from /workspace/CAP/:
#     bash eval/ablation/module/eval_all.sh

set -euo pipefail

VARIANTS="${VARIANTS:-no_hier no_algebraic no_cvae no_physics no_equivariance no_lipschitz}"
SEED="${SEED:-0}"
RUN_ROOT="${RUN_ROOT:-runs/module}"

# Eval data — A test set + B test set.
A_MANIFEST="${A_MANIFEST:-dataset/dataset_a/manifest.json}"
A_DATA="${A_DATA:-dataset/dataset_a/data}"
A_SPLIT="${A_SPLIT:-test_iid}"
B_MANIFEST="${B_MANIFEST:-dataset/dataset_b/manifest.json}"
B_DATA="${B_DATA:-dataset/dataset_b/data}"
B_SPLIT="${B_SPLIT:-test}"

# Optional: comma-separated task list for success_rate (depends on dataset_a layout).
TASKS="${TASKS:-open_drawer close_drawer pull_handle push_button rotate_knob}"

N_BATCHES="${N_BATCHES:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ ! -d eval ]]; then
    echo "ERROR: run from CAP root.  cwd=$(pwd)"
    exit 1
fi

# ─── Helper: run all eval scripts for one ckpt ─────────────────────────
run_eval_suite() {
    local CKPT="$1"
    local OUT="$2"
    local MANIFEST="$3"
    local DATA="$4"
    local SPLIT="$5"
    local DSFLAG="$6"          # "a" or "b"

    if [[ ! -f "$CKPT" ]]; then
        echo "    skip: ckpt missing → $CKPT"
        return
    fi
    mkdir -p "$OUT"

    # --enable-physics passes through to algebraic_gaps; harmless on no_physics
    # variants because executor short-circuits when enable_physics=False.
    python -m eval.algebraic_gaps \
        --ckpt "$CKPT" --output-dir "$OUT/algebraic_gaps" \
        --dataset "$DSFLAG" --manifest "$MANIFEST" --data-dir "$DATA" --split "$SPLIT" \
        --n-batches "$N_BATCHES" --batch-size "$BATCH_SIZE" \
        --enable-physics  || echo "      [algebraic_gaps] failed"

    python -m eval.trajectory_metrics \
        --ckpt "$CKPT" --output-dir "$OUT/trajectory_metrics" \
        --dataset "$DSFLAG" --manifest "$MANIFEST" --data-dir "$DATA" --split "$SPLIT" \
        --n-batches "$N_BATCHES" --batch-size "$BATCH_SIZE" \
        --enable-physics  || echo "      [trajectory_metrics] failed"

    python -m eval.success_rate \
        --ckpt "$CKPT" --output-dir "$OUT/success_rate" \
        --dataset "$DSFLAG" --manifest "$MANIFEST" --data-dir "$DATA" --split "$SPLIT" \
        --tasks $TASKS --zero-shot --enable-physics  \
        || echo "      [success_rate] failed"

    python -m eval.diversity \
        --ckpt "$CKPT" --output-dir "$OUT/diversity" \
        --dataset "$DSFLAG" --manifest "$MANIFEST" --data-dir "$DATA" --split "$SPLIT" \
        --num-samples 16  || echo "      [diversity] failed"
}

# ─── Loop ──────────────────────────────────────────────────────────────
for V in $VARIANTS; do
    A_CKPT="${RUN_ROOT}/${V}/seed_${SEED}/ckpt/main_exp_final.pt"
    B_CKPT="${RUN_ROOT}/${V}/seed_${SEED}/finetune_b/ckpt/main_exp_final.pt"

    echo ""
    echo "═══ eval ${V} (seed ${SEED}) ═══"

    echo "  → A ckpt"
    run_eval_suite "$A_CKPT" \
        "${RUN_ROOT}/${V}/seed_${SEED}/eval_a" \
        "$A_MANIFEST" "$A_DATA" "$A_SPLIT" "a"

    echo "  → B ckpt"
    run_eval_suite "$B_CKPT" \
        "${RUN_ROOT}/${V}/seed_${SEED}/eval_b" \
        "$B_MANIFEST" "$B_DATA" "$B_SPLIT" "b"
done

# ─── Run main model on ALL 3 seeds for mean±std baseline ──────────────
# Ablations use 1 seed per the standard protocol (Act4D MD: "主表 3 seeds，
# 其他 1 seed").  The main row in Tab 6 should be averaged over the 3
# pre-trained main seeds at runs/main_a/seed_{0,1,2}/ and
# runs/finetune_b/seed_{0,1,2}/.
MAIN_SEEDS="${MAIN_SEEDS:-0 1 2}"
MAIN_OUT="${MAIN_OUT:-runs/module/_main}"

for MS in $MAIN_SEEDS; do
    A_CK="runs/main_a/seed_${MS}/ckpt/main_exp_final.pt"
    B_CK="runs/finetune_b/seed_${MS}/ckpt/main_exp_final.pt"
    if [[ ! -f "$A_CK" ]]; then
        echo "  main seed=${MS} A ckpt not found ($A_CK), skipping"
        continue
    fi
    echo ""
    echo "═══ eval main (seed ${MS}) ═══"
    echo "  → A ckpt"
    run_eval_suite "$A_CK" "${MAIN_OUT}/seed_${MS}/eval_a" \
        "$A_MANIFEST" "$A_DATA" "$A_SPLIT" "a"
    if [[ -f "$B_CK" ]]; then
        echo "  → B ckpt"
        run_eval_suite "$B_CK" "${MAIN_OUT}/seed_${MS}/eval_b" \
            "$B_MANIFEST" "$B_DATA" "$B_SPLIT" "b"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All evals done."
echo "  Next: python eval/ablation/module/aggregate.py"
echo "═══════════════════════════════════════════════════════════════════"
