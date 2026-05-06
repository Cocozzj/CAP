#!/usr/bin/env bash
# train_a.sh — Train all module-ablation variants on Dataset-A.
#
# For each variant in variants.py: patch configs, run trainer.
# Output goes to runs/ablation/module/<variant>/seed_<S>/.
#
# Run from /workspace/CAP/:
#     bash eval/ablation/module/train_a.sh
#
# Override:
#     VARIANTS="no_physics no_algebraic" SEED=1 bash eval/ablation/module/train_a.sh
#     MAX_EPOCHS=80 bash eval/ablation/module/train_a.sh   # cheaper sweep

set -euo pipefail

# ─── Configurable knobs ─────────────────────────────────────────────────
VARIANTS="${VARIANTS:-no_hier no_algebraic no_cvae no_physics no_equivariance no_lipschitz}"
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

MANIFEST="${MANIFEST:-dataset/dataset_a/manifest.json}"
DATA_DIR="${DATA_DIR:-dataset/dataset_a/data}"

CFG_ROOT="${CFG_ROOT:-configs/_ablation}"
RUN_ROOT="${RUN_ROOT:-runs/ablation/module}"

# ─── Sanity ─────────────────────────────────────────────────────────────
if [[ ! -d configs ]] || [[ ! -d train ]] || [[ ! -d eval ]]; then
    echo "ERROR: run from CAP root (need configs/ train/ eval/ here). cwd=$(pwd)"
    exit 1
fi

# Hygiene — clear pycache so any code edits to trainer/loss take effect.
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

# ─── Loop ──────────────────────────────────────────────────────────────
for V in $VARIANTS; do
    OUT_DIR="${RUN_ROOT}/${V}/seed_${SEED}"
    CFG_DIR="${CFG_ROOT}/${V}"
    LOG="${OUT_DIR}/train_a.log"

    if [[ -f "${OUT_DIR}/ckpt/main_exp_final.pt" ]]; then
        echo ""
        echo "  [${V}] A ckpt exists, skipping (delete to re-run)"
        continue
    fi

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  Variant: ${V}    seed=${SEED}    out=${OUT_DIR}"
    echo "═══════════════════════════════════════════════════════════════════"

    # 1) Generate per-variant config files.
    python eval/ablation/module/make_config.py \
        --variant     "$V" \
        --base-config configs/config.yaml \
        --base-loss-a configs/loss.yaml \
        --base-loss-b configs/loss_b.yaml \
        --out-dir     "$CFG_DIR"

    # 2) Read trainer_flags.txt for any --no-physics / --no-kl-anneal.
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

    # 3) Train.
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

    echo "  ✔ ${V} A done. ckpt: ${OUT_DIR}/ckpt/main_exp_final.pt"
done

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  All A-side ablations trained."
echo "  Next: bash eval/ablation/module/finetune_b.sh"
echo "═══════════════════════════════════════════════════════════════════"
