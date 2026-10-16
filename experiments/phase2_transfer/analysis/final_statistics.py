import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("experiments")
OUT = BASE / "phase2_transfer/analysis"
OUT.mkdir(parents=True, exist_ok=True)

zero = pd.read_csv(
    BASE / "phase1_ood/results/zero_shot_topology_v2.csv"
)
adapt = pd.read_csv(
    BASE / "phase2_transfer/results/transfer_full.csv"
)

KEY = [
    "model", "train_family", "train_seed",
    "test_family", "path", "graph_seed"
]

assert len(zero) == 1200
assert len(adapt) == 10800

assert not zero.duplicated(KEY).any()
assert not adapt.duplicated(KEY + ["adapt_graphs", "steps"]).any()

assert set(adapt["adapt_graphs"]) == {1, 5, 20}
assert set(adapt["steps"]) == {10, 100, 1000}

zero = zero[KEY + [
    "edges", "cutrank50", "cutrank50_normalized",
    "cut05_normalized"
]].rename(columns={
    "edges": "zero_edges",
    "cutrank50": "zero_cut",
    "cutrank50_normalized": "zero_rank50",
    "cut05_normalized": "zero_05"
})

df = adapt.merge(
    zero,
    on=KEY,
    how="left",
    validate="many_to_one",
    indicator=True
)

assert len(df) == 10800
assert (df["_merge"] == "both").all()
assert (df["edges"] == df["zero_edges"]).all()

df["delta_rank50_pp"] = (
    df["cutrank50_normalized"] - df["zero_rank50"]
) * 100

df["delta_05_pp"] = (
    df["cut05_normalized"] - df["zero_05"]
) * 100

df["adapt_rank50_pct"] = df["cutrank50_normalized"] * 100
df["zero_rank50_pct"] = df["zero_rank50"] * 100

# Exact pairing integrity
assert df.groupby(
    ["model", "train_family", "train_seed",
     "test_family", "adapt_graphs", "steps"]
).size().eq(10).all()

df.drop(columns=["_merge"]).to_csv(
    OUT / "paired_transfer_detailed.csv",
    index=False
)

# Hierarchical paired bootstrap:
# resample training checkpoints (source family x seed),
# and independently resample target graphs within each family.
# Keep the same sampled units across models for paired comparisons.

RNG = np.random.default_rng(2026)
B = 3000

def paired_ci(sub, metric="delta_rank50_pp"):
    sources = sorted(sub["train_family"].unique())
    seeds = sorted(sub["train_seed"].unique())
    families = sorted(sub["test_family"].unique())

    # Shape: [source checkpoint, target family, graph]
    pivot = sub.pivot_table(
        index=["train_family", "train_seed"],
        columns=["test_family", "graph_seed"],
        values=metric,
        aggfunc="first"
    )

    checkpoints = [(s, k) for s in sources for k in seeds]
    graph_ids = {
        f: sorted(sub.loc[
            sub["test_family"] == f, "graph_seed"
        ].unique())
        for f in families
    }

    assert len(checkpoints) == 10
    assert all(len(graph_ids[f]) == 10 for f in families)

    arr = np.array([
        [
            [pivot.loc[ck, (f, g)] for g in graph_ids[f]]
            for f in families
        ]
        for ck in checkpoints
    ], dtype=float)

    assert arr.shape == (10, 4, 10)
    assert np.isfinite(arr).all()

    point = float(arr.mean())
    boots = np.empty(B)

    for b in range(B):
        ck_idx = RNG.integers(0, 10, 10)
        fam_means = []
        for fi in range(4):
            graph_idx = RNG.integers(0, 10, 10)
            fam_means.append(
                arr[ck_idx, fi][:, graph_idx].mean()
            )
        boots[b] = np.mean(fam_means)

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)

rows = []

for (model, graphs, steps), sub in df.groupby(
    ["model", "adapt_graphs", "steps"]
):
    point, lo, hi = paired_ci(sub)
    rows.append({
        "model": model,
        "adapt_graphs": graphs,
        "steps": steps,
        "zero_rank50_pct": sub["zero_rank50_pct"].mean(),
        "adapt_rank50_pct": sub["adapt_rank50_pct"].mean(),
        "delta_pp": point,
        "ci95_low_pp": lo,
        "ci95_high_pp": hi,
        "n": len(sub)
    })

summary = pd.DataFrame(rows).sort_values(
    ["model", "adapt_graphs", "steps"]
)

summary.to_csv(
    OUT / "paired_transfer_summary.csv",
    index=False
)

print("\n=== PAIRED TRANSFER IMPROVEMENT (pp) ===")
print(summary.round(3).to_string(index=False))

print("\n=== 20 GRAPHS / 1000 STEPS ===")
print(
    summary[
        (summary["adapt_graphs"] == 20) &
        (summary["steps"] == 1000)
    ].round(3).to_string(index=False)
)

print("\n=== IMPROVEMENT BY TARGET FAMILY ===")
family = (
    df[(df["adapt_graphs"] == 20) &
       (df["steps"] == 1000)]
    .groupby(["model", "test_family"])
    .agg(
        zero_pct=("zero_rank50_pct", "mean"),
        adapted_pct=("adapt_rank50_pct", "mean"),
        delta_pp=("delta_rank50_pp", "mean")
    )
    .reset_index()
)
print(family.round(3).to_string(index=False))
family.to_csv(
    OUT / "transfer_by_topology.csv",
    index=False
)

print("\n=== ER vs RRG SOURCE ===")
source = (
    df[(df["adapt_graphs"] == 20) &
       (df["steps"] == 1000)]
    .groupby(["model", "train_family"])
    .agg(
        zero_pct=("zero_rank50_pct", "mean"),
        adapted_pct=("adapt_rank50_pct", "mean"),
        delta_pp=("delta_rank50_pp", "mean")
    )
    .reset_index()
)
print(source.round(3).to_string(index=False))
source.to_csv(
    OUT / "transfer_by_source.csv",
    index=False
)

print("\nVALIDATION PASSED")
print("Zero-shot:", len(zero))
print("Transfer:", len(adapt))
print("Paired rows:", len(df))
print("Saved to:", OUT)
