#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

BASE="experiments/protocol2026"
CKPT="$BASE/checkpoints/production"
RESULT="$BASE/results/production"
LOG="$BASE/logs/production"

mkdir -p "$RESULT" "$LOG"

echo "Waiting for production training..."

while kill -0 51436 2>/dev/null; do
    sleep 30
done

echo "Training launcher finished."

python - <<'PY'
from pathlib import Path
import hashlib
import json
import torch

base = Path("experiments/protocol2026/checkpoints/production")
manifest = Path("dataset/protocol2026_v2/manifest.json")
digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

for model in ["pigcn_jk", "pigat_jk", "pigatv2_jk"]:
    for family in ["RRG", "ER"]:
        for seed in range(1, 6):
            path = base / f"{model}_{family}_seed{seed}.pt"
            if not path.is_file():
                raise RuntimeError(f"Missing: {path}")

            ckpt = torch.load(
                path, map_location="cpu", weights_only=False
            )
            args = ckpt["args"]

            assert args["model"] == model
            assert args["family"] == family
            assert int(args["seed"]) == seed
            assert args["feature_mode"] == "random"
            assert args["loss"] == "qubo"
            assert ckpt["manifest_sha256"] == digest
            assert ckpt["model_state"]
            assert 1 <= ckpt["best_step"] <= 10000

print("ALL 30 CHECKPOINTS VALIDATED")
PY

python -m experiments.protocol2026.scripts.eval_production \
    --device cuda \
    --output "$RESULT/zero_shot.csv" \
    > "$LOG/zero_shot.log" 2>&1

echo "ZERO-SHOT EVALUATION COMPLETED"
