import csv
import math
from pathlib import Path

from statsmodels.stats.multitest import multipletests

BASE = Path("experiments/protocol2026/results/production")
src = BASE / "adaptation_statistics.csv"
dst = BASE / "adaptation_statistics_holm.csv"

if not src.exists():
    print("Adaptation statistics not available yet.")
    raise SystemExit(0)

with src.open(newline="") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print("No statistical results yet.")
    raise SystemExit(0)

pvalues = [float(r["paired_t_pvalue"]) for r in rows]

if not all(math.isfinite(p) and 0 <= p <= 1 for p in pvalues):
    raise ValueError("Invalid p-values; check seed counts and variance.")

reject, corrected, _, _ = multipletests(
    pvalues,
    alpha=0.05,
    method="holm"
)

for r, p, significant in zip(rows, corrected, reject):
    r["holm_pvalue"] = p
    r["significant_005"] = bool(significant)

with dst.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("Saved:", dst)
print("Comparisons:", len(rows))
print("Significant after Holm:", sum(reject))
