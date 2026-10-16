import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("experiments/phase2_transfer/analysis")
df = pd.read_csv(OUT / "paired_transfer_detailed.csv")

df = df[
    (df["adapt_graphs"] == 20) &
    (df["steps"] == 1000)
]

key = ["train_family", "train_seed", "test_family", "graph_seed"]

pivot = df.pivot(
    index=key,
    columns="model",
    values="adapt_rank50_pct"
)

assert len(pivot) == 400
assert not pivot.isna().any().any()

pairs = [
    ("pigatv2_jk", "pigat_jk"),
    ("pigatv2_jk", "pigcn_jk"),
    ("pigcn_jk", "pigat_jk")
]

rng = np.random.default_rng(2026)
B = 5000
rows = []

for a, b in pairs:
    delta = (pivot[a] - pivot[b]).rename("delta").reset_index()

    sources = sorted(delta["train_family"].unique())
    seeds = sorted(delta["train_seed"].unique())
    families = sorted(delta["test_family"].unique())

    checkpoints = [(s, seed) for s in sources for seed in seeds]

    graph_ids = {
        f: sorted(delta.loc[
            delta["test_family"] == f, "graph_seed"
        ].unique())
        for f in families
    }

    lookup = delta.set_index(
        ["train_family", "train_seed", "test_family", "graph_seed"]
    )["delta"]

    arr = np.array([
        [
            [lookup.loc[(s, seed, f, g)] for g in graph_ids[f]]
            for f in families
        ]
        for s, seed in checkpoints
    ])

    assert arr.shape == (10, 4, 10)

    boot = np.empty(B)

    for i in range(B):
        ck = rng.integers(0, 10, 10)
        means = []

        for fi in range(4):
            graphs = rng.integers(0, 10, 10)
            means.append(arr[ck, fi][:, graphs].mean())

        boot[i] = np.mean(means)

    lo, hi = np.percentile(boot, [2.5, 97.5])

    rows.append({
        "comparison": f"{a} - {b}",
        "difference_pp": arr.mean(),
        "ci95_low_pp": lo,
        "ci95_high_pp": hi
    })

result = pd.DataFrame(rows)
result.to_csv(OUT / "model_comparison_ci.csv", index=False)

print("\n=== MODEL COMPARISON 95% CI ===")
print(result.round(4).to_string(index=False))
print("\nSAVED:", OUT / "model_comparison_ci.csv")
