#!/usr/bin/env bash
# train_sweep.sh — Train K-sweep variants on Dataset-A for Theorem 1 verification.
#
# Sweep K ∈ {64, 128, 256, 1024} sequentially at 80 ep.  K=512 reuses the
# existing main model (`runs/main_a/seed_0/`, 150 ep) — not re-trained.
# K=2048 skipped to save GPU-h; reintroduce by setting KS="64 128 256 1024 2048".
#
# Run from /workspace/CAP/ (or wherever CAP is checked out):
#     bash eval/ablation/ksweep/train_sweep.sh
#
# Override defaults with env vars:
#     KS="64 256 1024" SEED=1 bash eval/ablation/ksweep/train_sweep.sh
#     MAX_EPOCHS=80 bash eval/ablation/ksweep/train_sweep.sh   # short ablations

set -euo pipefail

# ─── Configurable knobs (env-var overridable) ─────────────────────────
KS="${KS:-64 128 256 1024}"            # K=512 reused from runs/main_a; K=2048 skipped
SEED="${SEED:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-80}"         # ablation default: 80 ep (vs main 150 ep)
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Dataset paths — assume A is at the conventional location.
MANIFEST="${MANIFEST:-dataset/dataset_a/manifest.json}"
DATA_DIR="${DATA_DIR:-dataset/dataset_a/data}"

# Where ksweep outputs go.  Flat under runs/ to match main_a / finetune_b
# siblings (instead of nested under runs/ablation/).
ROOT_OUT="${ROOT_OUT:-runs/ksweep}"
CFG_DIR="${CFG_DIR:-configs/_ksweep}"

# ─── Sanity: must run from CAP root (where configs/ and train/ live) ──
if [[ ! -d configs ]] || [[ ! -d train ]] || [[ ! -d eval ]]; then
    echo "ERROR: run from /workspace/CAP/ — couldn't see configs/ train/ eval/ here."
    echo "  cwd: $(pwd)"
    exit 1
fi

# ─── Hygiene: clear pycache so updated loss.py / stages.py take effect ──
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

mkdir -p "$ROOT_OUT" "$CFG_DIR"

# ─── Pretty banner ──────────────────────────────────────────────────────
echo "=================================================================="
echo "  K-sweep training"
echo "    K values   : $KS"
echo "    seed       : $SEED"
echo "    max_epochs : ${MAX_EPOCHS:-<curriculum default>}"
echo "    batch / GPU: $BATCH_SIZE  (× $NPROC_PER_NODE GPU)"
echo "    out root   : $ROOT_OUT"
echo "=================================================================="

# ─── Loop over K values ───────────────────────────────────────────────
for K in $KS; do
    OUT_DIR="${ROOT_OUT}/K${K}/seed_${SEED}"
    CFG_OUT="${CFG_DIR}/config_K${K}.yaml"
    LOG="${OUT_DIR}/train.log"

    if [[ -f "${OUT_DIR}/ckpt/main_exp_final.pt" ]]; then
        echo ""
        echo "  [K=${K}] ckpt already exists, skipping (delete to re-run)"
        echo "    ${OUT_DIR}/ckpt/main_exp_final.pt"
        continue
    fi

    echo ""
    echo "─────────── K=${K} ───────────"

    # 1) Patch base config to inject the requested K.
    python eval/ablation/ksweep/make_config.py \
        --base configs/config.yaml \
        --K    "$K" \
        --out  "$CFG_OUT"

    # 2) Train.  Stage preset 'a' = DEFAULT_STAGES (4-stage A curriculum, 150 ep).
    #    Optional --max-epochs caps each stage for cheaper sweeps.
    EXTRA_ARGS=()
    if [[ -n "$MAX_EPOCHS" ]]; then
        EXTRA_ARGS+=(--max-epochs "$MAX_EPOCHS")
    fi

    mkdir -p "$OUT_DIR"

    torchrun --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
        -m train.trainer \
        --config       "$CFG_OUT" \
        --loss-config  configs/loss.yaml \
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
        "${EXTRA_ARGS[@]}" \
        2>&1 | tee "$LOG"

    echo "  ✔ K=${K} done. ckpt: ${OUT_DIR}/ckpt/main_exp_final.pt"
done

echo ""
echo "=================================================================="
echo "  All K-variants trained.  Next: bash eval/ablation/ksweep/eval_sweep.sh"
echo "=================================================================="
