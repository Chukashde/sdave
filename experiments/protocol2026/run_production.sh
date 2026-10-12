#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CKPT="experiments/protocol2026/checkpoints/production"
LOG="experiments/protocol2026/logs/production"

mkdir -p "$CKPT" "$LOG"

for model in pigcn_jk pigat_jk pigatv2_jk; do
  for family in RRG ER; do
    for seed in 1 2 3 4 5; do

      name="${model}_${family}_seed${seed}"
      checkpoint="$CKPT/${name}.pt"

      if [[ -f "$checkpoint" ]]; then
        echo "SKIP $name"
        continue
      fi

      echo "START $name"

      python -m experiments.protocol2026.scripts.train_generalization \
        --production \
        --model "$model" \
        --family "$family" \
        --seed "$seed" \
        --out "$checkpoint" \
        > "$LOG/${name}.log" 2>&1

      echo "DONE $name"
    done
  done
done

echo "ALL 30 RUNS COMPLETED"
