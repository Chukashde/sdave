import pandas as pd
from pathlib import Path

root = Path("experiments/phase1_ood/results")
bls_dir = root / "bls_topology"

parts = []

for seed in range(1, 21):
    p = bls_dir / f"seed_{seed}.csv"
    assert p.exists(), f"Missing: {p}"

    df = pd.read_csv(p)
    assert len(df) == 40, f"Seed {seed}: {len(df)}/40"
    assert (df["bls_seed"] == seed).all()
    parts.append(df)

bls = pd.concat(parts, ignore_index=True)

assert len(bls) == 800
assert not bls.duplicated(["path", "bls_seed"]).any()

counts = bls.groupby("path")["bls_seed"].nunique()
assert len(counts) == 40
assert (counts == 20).all()

best = (
    bls.groupby(
        ["family", "path", "graph_seed", "n", "edges"],
        as_index=False
    )["cut"]
    .max()
    .rename(columns={"cut": "bls_best20"})
)

assert len(best) == 40
assert (best["bls_best20"] > 0).all()

best.to_csv(root / "topology_bls_best20.csv", index=False)

gnn = pd.read_csv(root / "zero_shot_topology_v2.csv")
assert len(gnn) == 1200

merged = gnn.merge(
    best[["path", "bls_best20"]],
    on="path",
    how="left",
    validate="many_to_one"
)

assert len(merged) == 1200
assert merged["bls_best20"].notna().all()

for name, col in [
    ("05", "cut05"),
    ("median", "cutmedian"),
    ("rank50", "cutrank50")
]:
    merged[f"ratio_{name}"] = merged[col] / merged["bls_best20"]

merged.to_csv(
    root / "topology_ood_cut_bls_detailed.csv",
    index=False
)

summary = (
    merged.groupby(["model", "train_family", "test_family"])
    .agg(
        count=("path", "size"),
        cut_bls05=("ratio_05", "mean"),
        cut_blsmedian=("ratio_median", "mean"),
        cut_blsrank50=("ratio_rank50", "mean")
    )
    .reset_index()
)

summary.to_csv(
    root / "topology_ood_cut_bls_summary.csv",
    index=False
)

print("\n=== TOPOLOGY OOD CUT/BLS (%) ===")
print(
    merged.groupby(["model", "test_family"])[
        ["ratio_05", "ratio_median", "ratio_rank50"]
    ].mean().mul(100).round(2).to_string()
)

print("\n=== OVERALL (%) ===")
print(
    merged.groupby("model")[
        ["ratio_05", "ratio_median", "ratio_rank50"]
    ].mean().mul(100).round(2).to_string()
)

print("\nVALIDATION PASSED")
print("BLS runs:", len(bls))
print("Graphs:", len(best))
print("GNN evaluations:", len(merged))
