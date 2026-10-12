import csv
from collections import defaultdict
from pathlib import Path

source = Path("experiments/protocol2026/results/bls_parallel")
output = Path("experiments/protocol2026/results/bls_reference_complete.csv")

groups = defaultdict(dict)

for file in sorted(source.glob("seed_*.csv")):
    with file.open(newline="") as f:
        for row in csv.DictReader(f):
            path = row["path"]
            seed = int(row["bls_seed"])
            cut = float(row["cut"])

            if seed in groups[path]:
                raise ValueError(f"Duplicate: {path}, seed={seed}")

            groups[path][seed] = {
                "cut": cut,
                "family": row["family"],
                "n": row["n"],
                "edges": row["edges"],
            }

complete = []

for path, seeds in sorted(groups.items()):
    if set(seeds) != set(range(1, 21)):
        continue

    best = max(v["cut"] for v in seeds.values())
    info = next(iter(seeds.values()))

    complete.append({
        "path": path,
        "family": info["family"],
        "n": info["n"],
        "edges": info["edges"],
        "reference_cut": best,
        "num_seeds": len(seeds),
    })

fields = [
    "path", "family", "n", "edges",
    "reference_cut", "num_seeds"
]

if output.exists():
    raise FileExistsError(f"Output already exists: {output}")

with output.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(complete)

print("Graphs with any BLS result:", len(groups))
print("Graphs with all 20 seeds:", len(complete))
print("Reference CSV:", output)
