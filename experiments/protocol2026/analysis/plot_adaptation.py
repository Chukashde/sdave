import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("experiments/protocol2026/results/production/adaptation")
OUT = Path("experiments/protocol2026/figures")
OUT.mkdir(parents=True, exist_ok=True)

# Зөвхөн final CSV файлуудыг уншина.
seed_data = defaultdict(list)

for file in sorted(BASE.glob("*.csv")):
    if file.name.startswith("."):
        continue

    with file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1200:
        print("SKIP:", file.name)
        continue

    for r in rows:
        key = (
            r["model"],
            r["train_family"],
            r["test_family"],
            r["train_seed"],
            r["regime"],
            int(r["steps"])
        )
        seed_data[key].append(float(r["cut_over_bls"]))

# Эхлээд graph-уудыг, дараа нь seed-үүдийг дундажлана.
grouped = defaultdict(list)

for key, values in seed_data.items():
    model, train, test, seed, regime, steps = key
    grouped[(model, train, test, regime, steps)].append(mean(values))

models = ["pigcn_jk", "pigat_jk", "pigatv2_jk"]

for model in models:
    for train in ("ER", "RRG"):
        for test in ("ER", "RRG"):
            fig, ax = plt.subplots(figsize=(7, 5))
            has_data = False

            for regime, label in [
                ("fine_tune", "Fine-tuning"),
                ("from_scratch", "From scratch")
            ]:
                steps_list = []
                scores = []

                for steps in (10, 100, 1000):
                    values = grouped.get(
                        (model, train, test, regime, steps), []
                    )
                    if values:
                        steps_list.append(steps)
                        scores.append(mean(values))

                if steps_list:
                    ax.plot(
                        steps_list, scores,
                        marker="o", linewidth=2, label=label
                    )
                    has_data = True

            if not has_data:
                plt.close(fig)
                continue

            ax.set_xscale("log")
            ax.set_xticks([10, 100, 1000])
            ax.set_xticklabels(["10", "100", "1000"])
            ax.set_xlabel("Optimization steps")
            ax.set_ylabel("Mean Cut / BLS Reference")
            ax.set_title(f"{model}: {train} → {test}")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()

            output = OUT / f"adaptation_{model}_{train}_to_{test}.png"
            fig.savefig(output, dpi=200)
            plt.close(fig)
            print("SAVED:", output)

print("\nNOTE: These plots are preliminary until 30/30 VALID.")
