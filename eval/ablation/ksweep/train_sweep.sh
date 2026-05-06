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
# Per-stage epochs for ablation: 25/20/20/35 = 100 total (vs main 35/35/25/55=150).
# Preserves curriculum shape: FULL is still the largest stage (35), reflecting
# its role as the "settle everything together" phase.  Override via env var or
# set STAGE_EPOCHS= (empty) + MAX_EPOCHS= (empty) for full main 150 ep.
STAGE_EPOCHS="${STAGE_EPOCHS:-25 20 20 35}"
MAX_EPOCHS="${MAX_EPOCHS:-}"           # empty by default; mutually exclusive with STAGE_EPOCHS
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"

# Dataset paths — assume A is at the conventional location.
MANIFEST="${MANIFEST:-dataset/dataset_a/manifest.json}"
DATA_DIR="${DATA_DIR:-dataset/dataset_a/data}"

# Where ksweep outputs go.  Flat under runs/ to match main_a / finetune_b
# siblings (instead of nested under runs/ablation/).  The patched config is
# saved alongside each K's ckpts so the run is fully self-contained.
ROOT_OUT="${ROOT_OUT:-runs/ksweep}"

# ─── Sanity: must run from CAP root (where configs/ and train/ live) ──
if [[ ! -d configs ]] || [[ ! -d train ]] || [[ ! -d eval ]]; then
    echo "ERROR: run from /workspace/CAP/ — couldn't see configs/ train/ eval/ here."
    echo "  cwd: $(pwd)"
    exit 1
fi

# ─── Hygiene: clear pycache so updated loss.py / stages.py take effect ──
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true

mkdir -p "$ROOT_OUT"

# ─── Pretty banner ──────────────────────────────────────────────────────
echo "=================================================================="
echo "  K-sweep training"
echo "    K values    : $KS"
echo "    seed        : $SEED"
echo "    stage_epochs: ${STAGE_EPOCHS:-<curriculum default>}"
echo "    max_epochs  : ${MAX_EPOCHS:-<unset>}"
echo "    batch / GPU : $BATCH_SIZE  (× $NPROC_PER_NODE GPU)"
echo "    out root    : $ROOT_OUT"
echo "=================================================================="

# ─── Loop over K values ───────────────────────────────────────────────
for K in $KS; do
    K_DIR="${ROOT_OUT}/K${K}"
    OUT_DIR="${K_DIR}/seed_${SEED}"
    CFG_OUT="${K_DIR}/config.yaml"          # K-specific config lives alongside ckpts
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
    #    --stage-epochs gives explicit per-stage budget; --max-epochs is the
    #    uniform cap fallback.  Mutually exclusive (trainer.py errors if both).
    EXTRA_ARGS=()
    if [[ -n "$STAGE_EPOCHS" ]]; then
        EXTRA_ARGS+=(--stage-epochs)
        for ep in $STAGE_EPOCHS; do EXTRA_ARGS+=("$ep"); done
    elif [[ -n "$MAX_EPOCHS" ]]; then
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
