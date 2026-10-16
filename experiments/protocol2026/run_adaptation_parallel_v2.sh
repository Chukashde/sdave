#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="experiments/protocol2026"
OUT="$BASE/results/production/adaptation"
LOG="$BASE/logs/adaptation"
VALIDATOR="$BASE/scripts/validate_adaptation_csv.py"

mkdir -p "$OUT" "$LOG"

run_one() {
    local model="$1"
    local family="$2"
    local seed="$3"
    local name="${model}_${family}_seed${seed}"
    local output="$OUT/${name}.csv"
    local tmp="$OUT/.${name}.running.csv"

    if [[ -f "$output" ]]; then
        if python "$VALIDATOR" "$output" "$model" "$family" "$seed"; then
            echo "SKIP $name"
            return 0
        fi

        echo "ERROR: Existing final CSV invalid: $output" >&2
        return 1
    fi

    if [[ -f "$tmp" ]]; then
        echo "PRESERVE incomplete $name"
        mv "$tmp" "${tmp}.incomplete.$(date +%s)"
    fi

    echo "START $name"

    if python -m experiments.protocol2026.scripts.eval_adaptation \
        --device cuda \
        --model "$model" \
        --family "$family" \
        --seed "$seed" \
        --steps 10 100 1000 \
        --output "$tmp" \
        > "$LOG/${name}.log" 2>&1
    then
        if python "$VALIDATOR" "$tmp" "$model" "$family" "$seed"; then
            mv "$tmp" "$output"
            echo "DONE $name"
        else
            echo "INVALID RESULT $name" >&2
            return 1
        fi
    else
        echo "FAILED $name" >&2
        return 1
    fi
}

export -f run_one
export BASE OUT LOG VALIDATOR

for model in pigcn_jk pigat_jk pigatv2_jk; do
    for family in RRG ER; do
        for seed in 1 2 3 4 5; do
            printf '%s %s %s\n' "$model" "$family" "$seed"
        done
    done
done | xargs -P 4 -n 3 bash -c 'run_one "$@"' _

echo "PARALLEL LAUNCHER FINISHED"
