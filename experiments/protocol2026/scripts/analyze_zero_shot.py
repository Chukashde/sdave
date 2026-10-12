import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

base = Path("experiments/protocol2026/results/production")
zero = base / "zero_shot.csv"
reference = base / "bls_test_reference_200.csv"
output = base / "zero_shot_summary.csv"

if output.exists():
    raise FileExistsError(output)

with reference.open(newline="") as f:
    refs = {r["path"]: float(r["reference_cut"]) for r in csv.DictReader(f)}

assert len(refs) == 200

with zero.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 6000, f"Expected 6000 rows, got {len(rows)}"

groups = defaultdict(lambda: defaultdict(list))

for r in rows:
    key = (
        r["model"],
        r["train_family"],
        r["test_family"],
        r["regime"],
    )
    seed = int(r["train_seed"])
    path = r["path"]

    assert path in refs
    assert refs[path] > 0

    ratio = float(r["cut05"]) / refs[path]
    groups[key][seed].append(ratio)

summary = []

for key, seeds in sorted(groups.items()):
    assert set(seeds) == set(range(1, 6))

    seed_means = []

    for seed in range(1, 6):
        assert len(seeds[seed]) == 100
        seed_means.append(np.mean(seeds[seed]))

    summary.append({
        "model": key[0],
        "train_family": key[1],
        "test_family": key[2],
        "regime": key[3],
        "mean_cut_over_bls": float(np.mean(seed_means)),
        "std_across_seeds": float(np.std(seed_means, ddof=1)),
        "num_seeds": 5,
        "graphs_per_seed": 100,
    })

with output.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("ZERO-SHOT ANALYSIS PASSED")
print("Summary rows:", len(summary))
print("Output:", output)

for r in summary:
    print(
        f'{r["model"]:12s} '
        f'{r["train_family"]}->{r["test_family"]} '
        f'{r["mean_cut_over_bls"]:.4f} '
        f'± {r["std_across_seeds"]:.4f}'
    )
