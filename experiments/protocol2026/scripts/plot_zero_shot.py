import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("experiments/protocol2026/results/production")
OUT = Path("experiments/protocol2026/figures")
OUT.mkdir(parents=True, exist_ok=True)

refs = {}
with open(BASE / "bls_test_reference_200.csv", newline="") as f:
    for r in csv.DictReader(f):
        refs[r["path"]] = float(r["reference_cut"])

groups = defaultdict(list)

with open(BASE / "zero_shot.csv", newline="") as f:
    for r in csv.DictReader(f):
        key = (
            r["model"],
            r["train_family"],
            r["test_family"],
            int(r["train_seed"]),
        )
        groups[key].append(
            float(r["cut05"]) / refs[r["path"]]
        )

seed_means = {
    key: sum(v) / len(v)
    for key, v in groups.items()
}

models = ["pigcn_jk", "pigat_jk", "pigatv2_jk"]
pairs = [
    ("ER", "ER"),
    ("ER", "RRG"),
    ("RRG", "ER"),
    ("RRG", "RRG"),
]

fig, ax = plt.subplots(figsize=(10, 5.5))

x = range(len(pairs))
width = 0.25

for i, model in enumerate(models):
    means = []
    errors = []

    for train, test in pairs:
        values = [
            seed_means[(model, train, test, seed)]
            for seed in range(1, 6)
        ]
        mean = sum(values) / len(values)
        sd = (
            sum((v - mean) ** 2 for v in values)
            / (len(values) - 1)
        ) ** 0.5

        means.append(mean)
        errors.append(sd)

    positions = [j + (i - 1) * width for j in x]

    ax.bar(
        positions,
        means,
        width,
        yerr=errors,
        capsize=3,
        label=model,
    )

ax.set_xticks(list(x))
ax.set_xticklabels([f"{a} → {b}" for a, b in pairs])
ax.set_ylabel("Cut / BLS best-known cut")
ax.set_title("Zero-shot Generalization (5 training seeds)")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()

output = OUT / "zero_shot_generalization.png"
fig.savefig(output, dpi=200)
plt.close(fig)

print("Saved:", output)
print("Seed groups:", len(seed_means))
