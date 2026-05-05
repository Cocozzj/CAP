#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_seeds.sh — launch seeded training runs for the main experiment.
#
# REQUIRED env vars (trainer's argparse rejects without them):
#   MANIFEST   path to data/dataset_<a|b>/manifest.json
#   DATA_DIR   path to data/dataset_<a|b>/data
#
# Usage:
#   MANIFEST=data/dataset_a/manifest.json DATA_DIR=data/dataset_a/data \
#     tools/run_seeds.sh runs/main_exp                # default: seeds 0..4
#
#   MANIFEST=... DATA_DIR=... \
#     tools/run_seeds.sh runs/main_exp 0 1 2 3 4      # explicit seed list
#
# Optional env vars:
#   DATASET           a (default) | b
#   STAGES_PRESET     a (default) | b — passed as --stages-preset
#                     a → 4-stage 150ep curriculum (RIGID/PLANNER/PHYSICS/FULL)
#                     b → single 30ep fine-tune stage (use with RESUME_TEMPLATE)
#   LOSS_CONFIG       path to loss yaml (default: configs/loss.yaml; for B
#                     fine-tune use configs/loss_b.yaml)
#   RESUME_TEMPLATE   per-seed init weights template, with literal "{SEED}"
#                     substituted per seed.  Example:
#                       RESUME_TEMPLATE="runs/main_exp/seed_{SEED}/ckpt/main_exp_final.pt"
#                     → seed 0 loads runs/main_exp/seed_0/ckpt/main_exp_final.pt
#                     → seed 1 loads runs/main_exp/seed_1/ckpt/main_exp_final.pt
#                     Used for B fine-tuning each seed from its corresponding
#                     A pre-trained checkpoint.
#   STAGE_DONE_FILE   filename to check for skip-if-finished (default
#                     stage_full_done.pt for A; for B set to stage_finetune_b_done.pt)
#   EXTRA_ARGS        forwarded verbatim to trainer (e.g. "--batch-size 16 --num-workers 8")
#   TORCHRUN_NPROC    >1 → launch via torchrun for DDP
#
# Each seed gets its own subdirectory:  $BASE/seed_${SEED}/
# Runs SEQUENTIALLY by default.  For parallel, drive from slurm/k8s with
# one machine per seed.
#
# Examples:
#   # Dataset-A 3-seed long training (default):
#   MANIFEST=dataset/dataset_a/manifest.json DATA_DIR=dataset/dataset_a/data \
#     TORCHRUN_NPROC=8 tools/run_seeds.sh runs/main_exp 0 1 2
#
#   # Dataset-B fine-tune from A's main_exp_final.pt:
#   MANIFEST=dataset/dataset_b/manifest.json DATA_DIR=dataset/dataset_b/data \
#     DATASET=b STAGES_PRESET=b LOSS_CONFIG=configs/loss_b.yaml \
#     RESUME_TEMPLATE="runs/main_exp/seed_{SEED}/ckpt/main_exp_final.pt" \
#     STAGE_DONE_FILE=stage_finetune_b_done.pt \
#     TORCHRUN_NPROC=8 tools/run_seeds.sh runs/finetune_b 0 1 2
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

# Required: manifest + data-dir.  Fail fast with a clear message instead of
# letting trainer's argparse print a less-friendly error per seed.
: "${MANIFEST:?Set MANIFEST=data/dataset_<a|b>/manifest.json before running}"
: "${DATA_DIR:?Set DATA_DIR=data/dataset_<a|b>/data before running}"
DATASET="${DATASET:-a}"

BASE="${1:-runs/main_exp}"
shift || true
if [[ $# -eq 0 ]]; then
    SEEDS=(0 1 2 3 4)
else
    SEEDS=("$@")
fi

# Anything in $EXTRA_ARGS is forwarded verbatim to trainer.
EXTRA_ARGS="${EXTRA_ARGS:-}"
STAGES_PRESET="${STAGES_PRESET:-a}"
LOSS_CONFIG="${LOSS_CONFIG:-configs/loss.yaml}"
RESUME_TEMPLATE="${RESUME_TEMPLATE:-}"
STAGE_DONE_FILE="${STAGE_DONE_FILE:-stage_full_done.pt}"

# Multi-GPU: set TORCHRUN_NPROC=N to launch via torchrun on N local GPUs.
# Multi-node: also set TORCHRUN_NNODES, TORCHRUN_NODE_RANK,
#             TORCHRUN_RDZV_ENDPOINT (host:port of node 0).
# Single-node port override: MASTER_PORT=NNNNN (default 29400).
TORCHRUN_NPROC="${TORCHRUN_NPROC:-1}"
if [[ "${TORCHRUN_NPROC}" -gt 1 ]]; then
    LAUNCH=("torchrun" "--nproc_per_node=${TORCHRUN_NPROC}")
    if [[ -n "${MASTER_PORT:-}" ]]; then
        LAUNCH+=("--master_port=${MASTER_PORT}")
    fi
    if [[ -n "${TORCHRUN_NNODES:-}" ]]; then
        LAUNCH+=("--nnodes=${TORCHRUN_NNODES}"
                 "--node_rank=${TORCHRUN_NODE_RANK:-0}"
                 "--rdzv_backend=c10d"
                 "--rdzv_endpoint=${TORCHRUN_RDZV_ENDPOINT}")
    fi
else
    LAUNCH=("python")
fi

mkdir -p "${BASE}"
echo "Base dir       : ${BASE}"
echo "Seeds          : ${SEEDS[*]}"
echo "Dataset        : ${DATASET}  (preset: ${STAGES_PRESET})"
echo "Loss config    : ${LOSS_CONFIG}"
echo "Resume tmpl    : ${RESUME_TEMPLATE:-<none>}"
echo "Skip-if-done   : ckpt/${STAGE_DONE_FILE}"
echo "Extra          : ${EXTRA_ARGS:-<none>}"
echo

for SEED in "${SEEDS[@]}"; do
    OUT="${BASE}/seed_${SEED}"
    if [[ -f "${OUT}/ckpt/${STAGE_DONE_FILE}" ]]; then
        echo "✓ seed=${SEED} already finished (${STAGE_DONE_FILE} exists) — skipping"
        continue
    fi
    echo "─────────────────────────────────────────────────────────────"
    echo "▶ seed=${SEED}  →  ${OUT}"

    # Per-seed init-weights checkpoint (only set if RESUME_TEMPLATE provided).
    # Substitute literal "{SEED}" in the template with the current seed number.
    RESUME_ARG=()
    if [[ -n "${RESUME_TEMPLATE}" ]]; then
        RESUME_CKPT="${RESUME_TEMPLATE//\{SEED\}/${SEED}}"
        if [[ -f "${RESUME_CKPT}" ]]; then
            RESUME_ARG=("--resume-from-ckpt" "${RESUME_CKPT}")
            echo "  ⟳ init from: ${RESUME_CKPT}"
        else
            # Fail fast — silently training from scratch on a short B
            # curriculum (30 epochs) would produce useless weights and waste
            # GPU hours.  If the source ckpt is missing, the user needs to
            # know NOW, not after the run finishes.
            echo "  ✗ ERROR: RESUME_TEMPLATE set but ckpt missing: ${RESUME_CKPT}"
            echo "    Either:"
            echo "      (a) wait for source training (e.g. Dataset-A) to finish"
            echo "      (b) unset RESUME_TEMPLATE if you really want from-scratch"
            exit 1
        fi
    fi
    echo "─────────────────────────────────────────────────────────────"

    "${LAUNCH[@]}" -m train.trainer \
        --config configs/config.yaml \
        --loss-config "${LOSS_CONFIG}" \
        --stages-preset "${STAGES_PRESET}" \
        --out-dir "${OUT}" \
        --seed "${SEED}" \
        --dataset "${DATASET}" \
        --manifest "${MANIFEST}" \
        --data-dir "${DATA_DIR}" \
        "${RESUME_ARG[@]}" \
        ${EXTRA_ARGS}
done

echo
echo "✓ All seeds complete.  Aggregate with:"
echo "    python tools/aggregate_seeds.py ${BASE} --eval-name algebraic_gaps"
