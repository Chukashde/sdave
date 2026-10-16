import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("experiments/protocol2026/results/production")
OUT = Path("experiments/protocol2026/figures")
OUT.mkdir(parents=True, exist_ok=True)

with open(BASE / "zero_shot_summary.csv", newline="") as f:
    rows = list(csv.DictReader(f))

models = ["pigcn_jk", "pigat_jk", "pigatv2_jk"]
labels = ["PI-GCN-JK", "PI-GAT-JK", "PI-GATv2-JK"]

for train, test in [
    ("ER", "ER"),
    ("ER", "RRG"),
    ("RRG", "ER"),
    ("RRG", "RRG")
]:
    values = []
    errors = []

    for model in models:
        match = [
            r for r in rows
            if r["model"] == model
            and r["train_family"] == train
            and r["test_family"] == test
        ]
        if len(match) != 1:
            raise ValueError(
                f"Expected one summary row: {model} {train}->{test}, "
                f"found {len(match)}"
            )

        r = match[0]
        values.append(float(r["mean_cut_over_bls"]))
        errors.append(float(r["std_across_seeds"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, yerr=errors, capsize=5)
    ax.set_ylabel("Mean Cut / BLS Reference")
    ax.set_title(f"Zero-shot Generalization: {train} → {test}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.2)

    for i, v in enumerate(values):
        ax.text(i, v + errors[i] + 0.008, f"{v:.4f}",
                ha="center", fontsize=10)

    fig.tight_layout()
    filename = OUT / f"zero_shot_{train}_to_{test}.png"
    fig.savefig(filename, dpi=200)
    plt.close(fig)
    print("SAVED:", filename)
