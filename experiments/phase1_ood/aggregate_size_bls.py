import pandas as pd
from pathlib import Path

root = Path("experiments/phase1_ood")
bls_dir = root / "results/bls_parallel"
out_dir = root / "results"

files = sorted(bls_dir.glob("seed_*.csv"))

assert len(files) == 20, f"Expected 20 seed files, found {len(files)}"

bls = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)

keys = ["path", "graph_seed", "n", "edges"]

assert len(bls) == 1200, f"Expected 1200 BLS rows, got {len(bls)}"
assert bls[keys + ["bls_seed"]].duplicated().sum() == 0
assert bls["bls_seed"].nunique() == 20
assert bls["path"].nunique() == 60
assert bls.groupby("path")["bls_seed"].nunique().eq(20).all()

best = (
    bls.groupby(keys, as_index=False)
    .agg(
        bls_best=("cut", "max"),
        bls_mean=("cut", "mean"),
        bls_std=("cut", "std")
    )
)

assert len(best) == 60

best.to_csv(
    out_dir / "size_bls_best20.csv",
    index=False
)

df = pd.read_csv(out_dir / "zero_shot_size.csv")

assert len(df) == 1800, f"Expected 1800 model rows, got {len(df)}"

merged = df.merge(
    best,
    on=keys,
    how="left",
    validate="many_to_one",
    indicator=True
)

assert merged["_merge"].eq("both").all()
assert merged["bls_best"].gt(0).all()
merged = merged.drop(columns="_merge")

for method, col in [
    ("cut05", "cut05"),
    ("cutmedian", "cutmedian"),
    ("cutrank50", "cutrank50")
]:
    merged[f"{method}_bls"] = merged[col] / merged["bls_best"]

merged.to_csv(
    out_dir / "size_ood_cut_bls_detailed.csv",
    index=False
)

summary = (
    merged.groupby(["model", "n"], as_index=False)
    .agg(
        cut05_bls=("cut05_bls", "mean"),
        cutmedian_bls=("cutmedian_bls", "mean"),
        cutrank50_bls=("cutrank50_bls", "mean"),
        samples=("path", "size")
    )
)

for col in ["cut05_bls", "cutmedian_bls", "cutrank50_bls"]:
    summary[col] *= 100

summary.to_csv(
    out_dir / "size_ood_cut_bls_summary.csv",
    index=False
)

print("\n=== SIZE OOD: CUT/BLS (%) ===")
print(summary.round(3).to_string(index=False))

print("\nSAVED:")
print(out_dir / "size_bls_best20.csv")
print(out_dir / "size_ood_cut_bls_detailed.csv")
print(out_dir / "size_ood_cut_bls_summary.csv")
