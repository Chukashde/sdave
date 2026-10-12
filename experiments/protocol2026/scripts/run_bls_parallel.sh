#!/usr/bin/env bash
set -euo pipefail

BASE="experiments/protocol2026"
OLD="$BASE/results/bls_v2_iid_20seeds.csv"
DIR="$BASE/results/bls_parallel"
LOG="$BASE/logs/bls_parallel"

mkdir -p "$DIR" "$LOG"

# Seed бүрт өмнөх CSV-ээс зөвхөн өөрийн мөрүүдийг хуулна.
# Ингэснээр --resume өмнө хийсэн ажлыг алгасна.
python - <<'PY'
import csv
from pathlib import Path

old = Path("experiments/protocol2026/results/bls_v2_iid_20seeds.csv")
outdir = Path("experiments/protocol2026/results/bls_parallel")

with old.open(newline="") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    assert fields is not None
    rows = list(reader)

for seed in range(1, 21):
    dst = outdir / f"seed_{seed}.csv"

    if dst.exists():
        print(f"EXISTS: {dst} (keeping existing progress)")
        continue

    selected = [r for r in rows if int(r["bls_seed"]) == seed]

    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    print(f"Seed {seed}: imported {len(selected)} existing runs")
PY

pids=()

for seed in $(seq 1 20); do
    echo "Starting seed $seed"

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u -m experiments.protocol2026.scripts.run_bls \
        --manifest dataset/protocol2026_v2/manifest.json \
        --split test_iid \
        --families RRG ER \
        --seeds "$seed" \
        --time_limit 30 \
        --output "$DIR/seed_${seed}.csv" \
        --resume \
        > "$LOG/seed_${seed}.log" 2>&1 &

    pids+=("$!")
done

failed=0

for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "FAILED process PID=$pid"
        failed=1
    fi
done

if [ "$failed" -eq 0 ]; then
    echo "ALL 20 BLS WORKERS COMPLETED"
else
    echo "Some workers failed. Inspect logs before resuming."
    exit 1
fi
