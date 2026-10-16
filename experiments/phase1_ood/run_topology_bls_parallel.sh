#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p experiments/phase1_ood/results/bls_topology
mkdir -p experiments/phase1_ood/logs/bls_topology

for seed in $(seq 1 20); do
    while [ "$(jobs -rp | wc -l)" -ge 8 ]; do
        wait -n
    done

    (
        OPENBLAS_NUM_THREADS=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        python -u -m experiments.protocol2026.scripts.run_bls \
            --manifest experiments/phase1_ood/data_topology_v2/manifest.json \
            --split test_topology_ood \
            --families SBM WS GEO SF \
            --seeds "$seed" \
            --time_limit 30 \
            --output "experiments/phase1_ood/results/bls_topology/seed_${seed}.csv" \
            --resume \
            > "experiments/phase1_ood/logs/bls_topology/seed_${seed}.log" 2>&1
    ) &
done

wait
echo "TOPOLOGY BLS FINISHED"
