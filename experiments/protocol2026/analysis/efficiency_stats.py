import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from scipy.stats import ttest_1samp, t

base = Path("experiments/protocol2026/results/production/adaptation")
data = defaultdict(dict)

for file in sorted(base.glob("*.csv")):
    with file.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (
                r["model"],
                r["train_family"],
                r["test_family"],
                int(r["train_seed"]),
                r["path"]
            )
            data[key][(r["regime"], int(r["steps"]))] = float(r["cut_over_bls"])

seed_diffs = defaultdict(list)

for key, values in data.items():
    ft = values[("fine_tune", 10)]
    fs = values[("from_scratch", 100)]
    seed_diffs[key[:4]].append(ft - fs)

groups = defaultdict(list)

for key, diffs in seed_diffs.items():
    model, train, test, seed = key
    groups[(model, train, test)].append(mean(diffs))

results = []

for key in sorted(groups):
    diffs = groups[key]
    assert len(diffs) == 5

    avg = mean(diffs)
    se = stdev(diffs) / (len(diffs) ** 0.5)
    ci = t.ppf(0.975, 4) * se
    p = ttest_1samp(diffs, 0).pvalue

    results.append((key, avg, avg-ci, avg+ci, p))

# Holm correction across 12 comparisons
order = sorted(range(len(results)), key=lambda i: results[i][4])
adjusted = [0.0] * len(results)
running = 0.0

for rank, i in enumerate(order):
    running = max(running, min(1.0, (12-rank) * results[i][4]))
    adjusted[i] = running

print("MODEL        TRANSFER   GAIN       95% CI             HOLM P")

for i, (key, avg, lo, hi, p) in enumerate(results):
    model, train, test = key
    print(
        f"{model:<12} {train+'->'+test:<9} "
        f"{avg:+.4f}   [{lo:+.4f}, {hi:+.4f}]   {adjusted[i]:.6f}"
    )

print("\nSignificant:", sum(p < 0.05 for p in adjusted), "/12")
