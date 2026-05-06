#!/usr/bin/env bash
# train_a.sh — Tier-2 loss-term ablations: A only (no B fine-tune).
#
# Run from /workspace/CAP/:
#     bash eval/ablation/loss/train_a.sh
#
# Output: runs/loss/<variant>/seed_<S>/
# Configs: configs/_ablation_loss/<variant>/

set -euo pipefail

VARIANTS="${VARIANTS:-no_L_clos no_L_inv no_L_comm no_L_hier no_L_nce no_kl_anneal}"
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29503}"

MANIFEST="${MANIFEST:-dataset/dataset_a/manifest.json}"
DATA_DIR="${DATA_DIR:-dataset/dataset_a/data}"

CFG_ROOT="${CFG_ROOT:-configs/_ablation_loss}"
RUN_ROOT="${RUN_ROOT:-runs/loss}"

if [[ ! -d configs ]] || [[ ! -d train ]] || [[ ! -d eval ]]; then
    echo "ERROR: run from CAP root.  cwd=$(pwd)"; exit 1
fi

find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

for V in $VARIANTS; do
    OUT_DIR="${RUN_ROOT}/${V}/seed_${SEED}"
    CFG_DIR="${CFG_ROOT}/${V}"
    LOG="${OUT_DIR}/train_a.log"

    if [[ -f "${OUT_DIR}/ckpt/main_exp_final.pt" ]]; then
        echo "  [${V}] ckpt exists, skipping"
        continue
    fi

    echo ""
    echo "═══ ${V}  seed=${SEED}  out=${OUT_DIR} ═══"

    python eval/ablation/loss/make_config.py \
        --variant     "$V" \
        --base-config configs/config.yaml \
        --base-loss-a configs/loss.yaml \
        --base-loss-b configs/loss_b.yaml \
        --out-dir     "$CFG_DIR"

    EXTRA_FLAGS=()
    if [[ -s "${CFG_DIR}/trainer_flags.txt" ]]; then
        while IFS= read -r line; do
            [[ -n "$line" ]] && EXTRA_FLAGS+=("$line")
        done < "${CFG_DIR}/trainer_flags.txt"
    fi
    if [[ -n "$MAX_EPOCHS" ]]; then
        EXTRA_FLAGS+=(--max-epochs "$MAX_EPOCHS")
    fi

    mkdir -p "$OUT_DIR"

    torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
        -m train.trainer \
        --config       "${CFG_DIR}/config.yaml" \
        --loss-config  "${CFG_DIR}/loss.yaml" \
        --stages-preset a \
        --dataset      a \
        --manifest     "$MANIFEST" \
        --data-dir     "$DATA_DIR" \
        --batch-size   "$BATCH_SIZE" \
        --num-workers  "$NUM_WORKERS" \
        --out-dir      "$OUT_DIR" \
        --seed         "$SEED" \
        --save-every   50 \
        --keep-last    1 \
        "${EXTRA_FLAGS[@]}" \
        2>&1 | tee "$LOG"

    echo "  ✔ ${V} done."
done

echo ""
echo "═══ All Tier 2 ablations trained.  Next: bash eval/ablation/loss/eval_all.sh ═══"
