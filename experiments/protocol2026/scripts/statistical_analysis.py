import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scipy import stats

BASE = Path("experiments/protocol2026/results/production")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ci95(values):
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan"), float("nan")

    sd = stats.tstd(values)
    margin = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return mean, mean - margin, mean + margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    path = BASE / "adaptation_seed_summary.csv"
    if not path.exists():
        print("Run analyze_adaptation.py first.")
        return

    rows = read_csv(path)

    # One observation per training seed and comparison.
    grouped = defaultdict(dict)

    for r in rows:
        key = (
            r["model"],
            r["train_family"],
            r["test_family"],
            int(r["steps"]),
        )
        seed = int(r["seed"])
        regime = r["regime"]

        if regime in grouped[key][seed] if seed in grouped[key] else False:
            raise RuntimeError(f"Duplicate row: {key}, {seed}, {regime}")

        grouped[key].setdefault(seed, {})[regime] = float(
            r["mean_ratio"]
        )

    results = []

    for key, seed_map in sorted(grouped.items()):
        model, train_family, test_family, steps = key

        paired = []
        for seed, values in sorted(seed_map.items()):
            if "fine_tune" in values and "from_scratch" in values:
                paired.append(
                    values["fine_tune"] - values["from_scratch"]
                )

        if len(paired) != 5 and not args.allow_partial:
            continue

        if len(paired) < 2:
            continue

        mean_delta, low, high = ci95(paired)

        test = stats.ttest_1samp(paired, popmean=0)

        results.append({
            "model": model,
            "train_family": train_family,
            "test_family": test_family,
            "steps": steps,
            "n_seeds": len(paired),
            "fine_minus_scratch": mean_delta,
            "ci95_low": low,
            "ci95_high": high,
            "paired_t_pvalue": float(test.pvalue),
        })

    if not results:
        print("Not enough completed paired seeds yet.")
        return

    output = BASE / "adaptation_statistics.csv"

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"STATISTICAL ANALYSIS PASSED: {len(results)} rows")
    print("Output:", output)

    for r in results:
        print(
            f'{r["model"]} '
            f'{r["train_family"]}->{r["test_family"]} '
            f'{r["steps"]} steps | '
            f'delta={r["fine_minus_scratch"]:+.4f} | '
            f'95% CI=[{r["ci95_low"]:.4f}, '
            f'{r["ci95_high"]:.4f}] | '
            f'p={r["paired_t_pvalue"]:.4g}'
        )


if __name__ == "__main__":
    main()
