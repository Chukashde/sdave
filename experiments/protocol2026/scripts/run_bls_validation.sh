#!/usr/bin/env bash
set -euo pipefail

BASE="experiments/protocol2026"
DIR="$BASE/results/bls_validation/full"
LOG="$BASE/logs/bls_validation"
mkdir -p "$DIR" "$LOG"

MAX_WORKERS=8
running=0
failed=0

for seed in $(seq 1 20); do
    (
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -u -m experiments.protocol2026.scripts.run_bls \
            --manifest dataset/protocol2026_v2/manifest.json \
            --split val \
            --families RRG ER \
            --seeds "$seed" \
            --time_limit 30 \
            --output "$DIR/seed_${seed}.csv" \
            --resume \
            > "$LOG/seed_${seed}.log" 2>&1
    ) &

    running=$((running + 1))

    if (( running >= MAX_WORKERS )); then
        wait -n || failed=1
        running=$((running - 1))
    fi
done

wait || failed=1

if (( failed != 0 )); then
    echo "WARNING: At least one worker failed; inspect logs"
    exit 1
fi

echo "VALIDATION BLS WORKERS FINISHED"
