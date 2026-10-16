import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, stdev
from scipy.stats import ttest_1samp, t

BASE = Path("experiments/protocol2026/results/production/adaptation")
OUT = Path("experiments/protocol2026/results/production/adaptation_paired_stats.csv")

scores = defaultdict(list)

for file in sorted(BASE.glob("*.csv")):
    with file.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 1200:
        raise ValueError(f"Invalid row count: {file}")

    for r in rows:
        key = (
            r["model"],
            r["train_family"],
            r["test_family"],
            int(r["train_seed"]),
            r["regime"],
            int(r["steps"])
        )
        scores[key].append(float(r["cut_over_bls"]))

seed_means = {k: mean(v) for k, v in scores.items()}
results = []

for model in ("pigcn_jk", "pigat_jk", "pigatv2_jk"):
    for train in ("ER", "RRG"):
        for test in ("ER", "RRG"):
            for steps in (10, 100, 1000):
                diffs = []

                for seed in range(1, 6):
                    base = (model, train, test, seed)
                    ft = seed_means[base + ("fine_tune", steps)]
                    fs = seed_means[base + ("from_scratch", steps)]
                    diffs.append(ft - fs)

                n = len(diffs)
                avg = mean(diffs)
                sd = stdev(diffs)
                se = sd / n**0.5
                critical = t.ppf(0.975, n - 1)
                p = ttest_1samp(diffs, 0).pvalue

                results.append({
                    "model": model,
                    "train": train,
                    "test": test,
                    "steps": steps,
                    "mean_gain": avg,
                    "ci_low": avg - critical * se,
                    "ci_high": avg + critical * se,
                    "p_raw": p,
                    "p_holm": None,
                    "n_seeds": n
                })

# Holm correction across all 36 comparisons.
order = sorted(range(len(results)), key=lambda i: results[i]["p_raw"])
adjusted = [0.0] * len(results)
running = 0.0
m = len(results)

for rank, i in enumerate(order):
    value = min(1.0, (m - rank) * results[i]["p_raw"])
    running = max(running, value)
    adjusted[i] = running

for i, r in enumerate(results):
    r["p_holm"] = adjusted[i]

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("MODEL        TRAIN TEST STEPS    GAIN      95% CI             HOLM P")
for r in results:
    print(
        f"{r['model']:<12} {r['train']:<5} {r['test']:<4} "
        f"{r['steps']:>4}  {r['mean_gain']:+.4f}  "
        f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
        f"{r['p_holm']:.6f}"
    )

print("\nHolm significant (p < 0.05):",
      sum(r["p_holm"] < 0.05 for r in results), "/ 36")
print("Saved:", OUT)
