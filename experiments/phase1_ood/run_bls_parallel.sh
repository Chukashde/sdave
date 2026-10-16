#!/usr/bin/env bash
set -euo pipefail

BASE="experiments/phase1_ood"
OUT="$BASE/results/bls_parallel"
LOG="$BASE/logs/bls_parallel"
MANIFEST="$BASE/data/manifest.json"

mkdir -p "$OUT" "$LOG"

run_seed() {
    local seed="$1"
    local file="$OUT/seed_${seed}.csv"
    local logfile="$LOG/seed_${seed}.log"

    local resume=()
    if [ -f "$file" ]; then
        resume=(--resume)
    fi

    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONPATH=. python -u -m experiments.protocol2026.scripts.run_bls \
        --manifest "$MANIFEST" \
        --split test_size_ood \
        --families ER RRG \
        --seeds "$seed" \
        --time_limit 30 \
        --output "$file" \
        "${resume[@]}" > "$logfile" 2>&1
}

export -f run_seed

# 12 workers at a time; wait for each batch
for start in 1 13; do
    pids=()

    end=$((start + 11))
    if [ "$end" -gt 20 ]; then end=20; fi

    for seed in $(seq "$start" "$end"); do
        run_seed "$seed" &
        pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "FAILED worker PID=$pid"
            failed=1
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "Batch failed; inspect logs before resuming."
        exit 1
    fi
done

echo "ALL 20 BLS SEEDS COMPLETED"
