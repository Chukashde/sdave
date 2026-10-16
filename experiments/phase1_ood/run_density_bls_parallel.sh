#!/usr/bin/env bash
set -euo pipefail

cd ~/projects/sdave

ROOT="experiments/phase1_ood"
OUT="$ROOT/results/bls_density"
LOG="$ROOT/logs/bls_density"

mkdir -p "$OUT" "$LOG"

# Existing launcher-ийг давхар ажиллуулахаас хамгаална.
exec 9>"$LOG/launcher.lock"
flock -n 9 || {
    echo "Density BLS launcher already running"
    exit 1
}

run_seed() {
    local seed="$1"

    echo "START seed=$seed"

    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    python -u -m experiments.protocol2026.scripts.run_bls \
        --manifest "$ROOT/data_density/manifest.json" \
        --split test_degree_ood \
        --families ER RRG \
        --seeds "$seed" \
        --time_limit 30 \
        --output "$OUT/seed_${seed}.csv" \
        --resume \
        > "$LOG/seed_${seed}.log" 2>&1

    echo "DONE seed=$seed"
}

export -f run_seed
export ROOT OUT LOG

# First batch: 12 workers
for seed in $(seq 1 12); do
    run_seed "$seed" &
done

wait

# Second batch: 8 workers
for seed in $(seq 13 20); do
    run_seed "$seed" &
done

wait

echo "Density BLS batches finished"
