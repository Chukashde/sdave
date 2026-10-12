#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="experiments/protocol2026"
OUT="$BASE/results/production/adaptation"
LOG="$BASE/logs/adaptation"

mkdir -p "$OUT" "$LOG"

models=(pigcn_jk pigat_jk pigatv2_jk)
families=(RRG ER)

for model in "${models[@]}"; do
  for family in "${families[@]}"; do
    for seed in 1 2 3 4 5; do
      name="${model}_${family}_seed${seed}"
      output="$OUT/${name}.csv"

      tmp="$OUT/.${name}.running.csv"

      if [[ -f "$output" ]]; then
        count=$(python - "$output" "$model" "$family" "$seed" <<'CHECK'
import csv
import sys

path, model, family, seed = sys.argv[1:]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))

expected = {
    (regime, step)
    for regime in ("fine_tune", "from_scratch")
    for step in (10, 100, 1000)
}

valid = (
    len(rows) == 1200
    and all(
        r["model"] == model
        and r["train_family"] == family
        and int(r["train_seed"]) == int(seed)
        for r in rows
    )
)

if valid:
    groups = {}
    for r in rows:
        groups.setdefault(r["path"], set()).add(
            (r["regime"], int(r["steps"]))
        )
    valid = len(groups) == 200 and all(
        v == expected for v in groups.values()
    )

print("VALID" if valid else "INCOMPLETE")
CHECK
)
        if [[ "$count" == "VALID" ]]; then
          echo "SKIP $name (1200 verified rows)"
          continue
        fi

        echo "INCOMPLETE $name: preserving old output"
        mv "$output" "$output.incomplete.$(date +%s)"
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
    done
  done
done

echo "ADAPTATION COMPLETED"
