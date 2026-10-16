import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev

BASE = Path("experiments/protocol2026/results/production")
ADAPT = BASE / "adaptation"

files = sorted(ADAPT.glob("*.csv"))
files = [p for p in files if not p.name.startswith(".")]

expected = {
    (regime, steps)
    for regime in ("fine_tune", "from_scratch")
    for steps in (10, 100, 1000)
}

rows = []
valid_files = []

for path in files:
    try:
        with path.open(newline="") as f:
            data = list(csv.DictReader(f))

        assert len(data) == 1200

        groups = defaultdict(set)

        for r in data:
            key = (
                r["model"],
                r["train_family"],
                r["train_seed"],
                r["path"]
            )
            groups[key].add((r["regime"], int(r["steps"])))

        assert len(groups) == 200
        assert all(v == expected for v in groups.values())

        rows.extend(data)
        valid_files.append(path.name)

    except Exception as e:
        print("SKIP INVALID:", path.name, e)

print(f"\nVALID FILES: {len(valid_files)}/30")
print(f"TOTAL ROWS: {len(rows)}")

# First average over graphs within each seed.
seed_groups = defaultdict(list)

for r in rows:
    key = (
        r["model"],
        r["train_family"],
        r["test_family"],
        int(r["train_seed"]),
        r["regime"],
        int(r["steps"])
    )
    seed_groups[key].append(float(r["cut_over_bls"]))

seed_means = {
    key: mean(values)
    for key, values in seed_groups.items()
}

# Aggregate over training seeds.
summary = defaultdict(list)

for key, value in seed_means.items():
    model, train, test, seed, regime, steps = key
    summary[(model, train, test, regime, steps)].append(value)

print("\n=== ADAPTATION SUMMARY ===")
print(
    f"{'MODEL':<13} {'TRAIN':<6} {'TEST':<6} "
    f"{'REGIME':<14} {'STEP':>5} {'MEAN':>9} {'STD':>9} {'N':>3}"
)

for key in sorted(summary):
    model, train, test, regime, steps = key
    values = summary[key]
    sd = stdev(values) if len(values) > 1 else float("nan")

    print(
        f"{model:<13} {train:<6} {test:<6} "
        f"{regime:<14} {steps:>5} "
        f"{mean(values):>9.4f} {sd:>9.4f} {len(values):>3}"
    )

print("\n=== FINE-TUNE MINUS FROM-SCRATCH ===")

for model in ("pigcn_jk", "pigat_jk", "pigatv2_jk"):
    for train in ("ER", "RRG"):
        for test in ("ER", "RRG"):
            for steps in (10, 100, 1000):
                diffs = []

                for seed in range(1, 6):
                    a = seed_means.get(
                        (model, train, test, seed, "fine_tune", steps)
                    )
                    b = seed_means.get(
                        (model, train, test, seed, "from_scratch", steps)
                    )

                    if a is not None and b is not None:
                        diffs.append(a - b)

                if diffs:
                    print(
                        f"{model:<13} {train}->{test:<3} "
                        f"steps={steps:<4} "
                        f"gain={mean(diffs):+.4f} "
                        f"seeds={len(diffs)}/5"
                    )
