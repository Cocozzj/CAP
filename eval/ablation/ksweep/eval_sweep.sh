#!/usr/bin/env bash
# eval_sweep.sh — Run eval.k_scaling_sweep over all K-variant ckpts.
#
# Picks up:
#   - K ∈ {64, 128, 256, 1024, 2048} from runs/ablation/ksweep/K{K}/seed_{S}/...
#   - K=512 from the original main run runs/main_exp/seed_{S}/...
#
# Run from /workspace/CAP/:
#     bash eval/ablation/ksweep/eval_sweep.sh
#
# Override:
#     KS="64 256 1024 2048" SEED=0 bash eval/ablation/ksweep/eval_sweep.sh

set -euo pipefail

KS="${KS:-64 128 256 512 1024 2048}"   # include 512 (main model) by default
SEED="${SEED:-0}"
N_BATCHES="${N_BATCHES:-16}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SPLIT="${SPLIT:-test_iid}"

# Where main K=512 model lives, vs ablation K-variants.
MAIN_DIR="${MAIN_DIR:-runs/main_exp}"
SWEEP_DIR="${SWEEP_DIR:-runs/ablation/ksweep}"

# Dataset paths (must point to A — closure / inverse / commutator are
# defined w.r.t. A's algebraic structure).
MANIFEST="${MANIFEST:-dataset/dataset_a/manifest.json}"
DATA_DIR="${DATA_DIR:-dataset/dataset_a/data}"

OUT_DIR="${OUT_DIR:-${SWEEP_DIR}/_eval}"
mkdir -p "$OUT_DIR"

# ─── Resolve ckpt paths ───────────────────────────────────────────────
declare -a CKPTS
declare -a CONFIGS

for K in $KS; do
    if [[ "$K" == "512" ]]; then
        # Main model lives elsewhere, with the un-patched config.
        CK="${MAIN_DIR}/seed_${SEED}/ckpt/main_exp_final.pt"
        CFG="configs/config.yaml"
    else
        CK="${SWEEP_DIR}/K${K}/seed_${SEED}/ckpt/main_exp_final.pt"
        CFG="configs/_ksweep/config_K${K}.yaml"
    fi
    if [[ ! -f "$CK" ]]; then
        echo "WARN: K=${K} ckpt missing: $CK  — skipping"
        continue
    fi
    if [[ ! -f "$CFG" ]]; then
        echo "WARN: K=${K} config missing: $CFG  — skipping"
        continue
    fi
    CKPTS+=("$CK")
    CONFIGS+=("$CFG")
    echo "  [K=${K}]  ckpt=$CK"
done

if [[ "${#CKPTS[@]}" -lt 2 ]]; then
    echo "ERROR: need ≥ 2 K-variant ckpts to fit a power law; only found ${#CKPTS[@]}"
    exit 1
fi

# ─── Run k_scaling_sweep ──────────────────────────────────────────────
echo ""
echo "=================================================================="
echo "  Running eval.k_scaling_sweep on ${#CKPTS[@]} ckpts"
echo "    out: $OUT_DIR"
echo "=================================================================="

python -m eval.k_scaling_sweep \
    --ckpts   "${CKPTS[@]}" \
    --configs "${CONFIGS[@]}" \
    --n-batches  "$N_BATCHES" \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUT_DIR" \
    --manifest   "$MANIFEST" \
    --data-dir   "$DATA_DIR" \
    --split      "$SPLIT"

echo ""
echo "  ✔ eval done."
echo "    summary : $OUT_DIR/summary.json"
echo "    points  : $OUT_DIR/points.csv"
echo "    fits    : $OUT_DIR/fit.json"
echo ""
echo "  Next: python eval/ablation/ksweep/plot_theorem1.py --summary $OUT_DIR/summary.json"
