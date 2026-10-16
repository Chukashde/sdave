#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="experiments/protocol2026"
OUT="$BASE/results/production/adaptation"
VALIDATOR="$BASE/scripts/validate_adaptation_csv.py"

for pid in 58189 58192 58194 58195; do
    while true; do
        state=$(ps -o stat= -p "$pid" 2>/dev/null || true)
        if [[ -z "$state" || "$state" == Z* ]]; then
            break
        fi
        sleep 15
    done
done

for job in \
    "pigcn_jk RRG 4" \
    "pigcn_jk RRG 5" \
    "pigcn_jk ER 1" \
    "pigcn_jk ER 2"
do
    read -r model family seed <<< "$job"
    name="${model}_${family}_seed${seed}"
    tmp="$OUT/.${name}.running.csv"
    final="$OUT/${name}.csv"

    if [[ -f "$final" ]]; then
        python "$VALIDATOR" "$final" "$model" "$family" "$seed"
    elif [[ -f "$tmp" ]] && python "$VALIDATOR" "$tmp" "$model" "$family" "$seed"; then
        mv "$tmp" "$final"
        echo "SAVED $name"
    else
        echo "INCOMPLETE $name — will retry"
    fi
done

bash "$BASE/run_adaptation_parallel_v2.sh"
