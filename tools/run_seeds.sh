#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_seeds.sh — launch 5 seeded training runs for the main experiment.
#
# Usage:
#   tools/run_seeds.sh runs/main_exp           # default: seeds 0..4, sequential
#   tools/run_seeds.sh runs/main_exp 0 1 2 3 4 # explicit seed list
#
# Each seed gets its own subdirectory:  $BASE/seed_${SEED}/
#
# Runs SEQUENTIALLY by default — one full curriculum per GPU box.
# To launch in parallel across machines, drive this from your scheduler
# (slurm/k8s) with one machine per seed.
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

BASE="${1:-runs/main_exp}"
shift || true
if [[ $# -eq 0 ]]; then
    SEEDS=(0 1 2 3 4)
else
    SEEDS=("$@")
fi

# Read remaining arguments — anything after the seeds list is forwarded
# to training.py.  Default is single-GPU; switch to torchrun for DDP.
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

    "${LAUNCH[@]}" training.py \
        --config configs/config.yaml \
        --loss-config configs/loss.yaml \
        --out-dir "${OUT}" \
        --seed "${SEED}" \
        ${EXTRA_ARGS}
done

echo
echo "✓ All seeds complete.  Aggregate with:"
echo "    python tools/aggregate_seeds.py ${BASE} --eval-name algebraic_gaps"
