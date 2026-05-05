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
#   EXTRA_ARGS        forwarded verbatim to trainer (e.g. "--auto-test --batch-size 16")
#   TORCHRUN_NPROC    >1 → launch via torchrun for DDP
#
# Each seed gets its own subdirectory:  $BASE/seed_${SEED}/
# Runs SEQUENTIALLY by default.  For parallel, drive from slurm/k8s with
# one machine per seed.
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

# Multi-GPU: set TORCHRUN_NPROC=N to launch via torchrun on N local GPUs.
# Multi-node: also set TORCHRUN_NNODES, TORCHRUN_NODE_RANK,
#             TORCHRUN_RDZV_ENDPOINT (host:port of node 0).
TORCHRUN_NPROC="${TORCHRUN_NPROC:-1}"
if [[ "${TORCHRUN_NPROC}" -gt 1 ]]; then
    LAUNCH=("torchrun" "--nproc_per_node=${TORCHRUN_NPROC}")
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
echo "Base dir : ${BASE}"
echo "Seeds    : ${SEEDS[*]}"
echo "Extra    : ${EXTRA_ARGS:-<none>}"
echo

for SEED in "${SEEDS[@]}"; do
    OUT="${BASE}/seed_${SEED}"
    if [[ -f "${OUT}/ckpt/stage_full_done.pt" ]]; then
        echo "✓ seed=${SEED} already finished (stage_full_done.pt exists) — skipping"
        continue
    fi
    echo "─────────────────────────────────────────────────────────────"
    echo "▶ seed=${SEED}  →  ${OUT}"
    echo "─────────────────────────────────────────────────────────────"

    "${LAUNCH[@]}" -m train.trainer \
        --config configs/config.yaml \
        --loss-config configs/loss.yaml \
        --out-dir "${OUT}" \
        --seed "${SEED}" \
        --dataset "${DATASET}" \
        --manifest "${MANIFEST}" \
        --data-dir "${DATA_DIR}" \
        ${EXTRA_ARGS}
done

echo
echo "✓ All seeds complete.  Aggregate with:"
echo "    python tools/aggregate_seeds.py ${BASE} --eval-name algebraic_gaps"
