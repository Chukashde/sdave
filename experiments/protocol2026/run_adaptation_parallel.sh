#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="experiments/protocol2026"
OUT="$BASE/results/production/adaptation"
LOG="$BASE/logs/adaptation"
mkdir -p "$OUT" "$LOG"

run_one() {
    local model="$1"
    local family="$2"
    local seed="$3"
    local name="${model}_${family}_seed${seed}"
    local output="$OUT/${name}.csv"
    local tmp="$OUT/.${name}.running.csv"

    if [[ -f "$output" ]]; then
        if python - "$output" "$model" "$family" "$seed" <<'PY'
import csv, sys
path, model, family, seed = sys.argv[1:]
try:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(r, s) for r in ("fine_tune", "from_scratch")
                         for s in (10, 100, 1000)}
    groups = {}
    for row in rows:
        assert row["model"] == model
        assert row["train_family"] == family
        assert int(row["train_seed"]) == int(seed)
        groups.setdefault(row["path"], set()).add(
            (row["regime"], int(row["steps"]))
        )
    assert len(rows) == 1200
    assert len(groups) == 200
    assert all(v == expected for v in groups.values())
except Exception:
    sys.exit(1)
PY
        then
            echo "SKIP $name"
            return 0
        fi
        echo "ERROR: invalid completed output $output" >&2
        return 1
    fi

    if [[ -f "$tmp" ]]; then
        mv "$tmp" "$tmp.incomplete.$(date +%s)"
    fi

    echo "START $name"

    python -m experiments.protocol2026.scripts.eval_adaptation \
        --device cuda \
        --model "$model" \
        --family "$family" \
        --seed "$seed" \
        --steps 10 100 1000 \
        --output "$tmp" \
        > "$LOG/${name}.log" 2>&1

    mv "$tmp" "$output"
    echo "DONE $name"
}

export -f run_one
export BASE OUT LOG

for model in pigcn_jk pigat_jk pigatv2_jk; do
    for family in RRG ER; do
        for seed in 1 2 3 4 5; do
            printf '%s %s %s\n' "$model" "$family" "$seed"
        done
    done
done | xargs -P 4 -n 3 bash -c 'run_one "$@"' _

echo "PARALLEL ADAPTATION COMPLETED"
