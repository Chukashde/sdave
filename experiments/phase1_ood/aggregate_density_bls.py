import pandas as pd
from pathlib import Path

ROOT = Path("experiments/phase1_ood")
BLS_DIR = ROOT / "results/bls_density"
ZERO = ROOT / "results/zero_shot_density.csv"
OUT = ROOT / "results"

frames = []

for seed in range(1, 21):
    path = BLS_DIR / f"seed_{seed}.csv"

    if not path.exists():
        raise RuntimeError(f"Missing: {path}")

    df = pd.read_csv(path)

    if len(df) != 80:
        raise RuntimeError(
            f"Seed {seed}: {len(df)}/80 runs completed"
        )

    if not (df["bls_seed"] == seed).all():
        raise RuntimeError(f"Wrong BLS seed in {path}")

    frames.append(df)

bls = pd.concat(frames, ignore_index=True)

assert len(bls) == 1600
assert bls["path"].nunique() == 80
assert not bls.duplicated(["path", "bls_seed"]).any()
assert bls.groupby("path")["bls_seed"].nunique().eq(20).all()

best = (
    bls.groupby("path", as_index=False)
    .agg(
        bls_best20=("cut", "max"),
        bls_mean20=("cut", "mean"),
        bls_std20=("cut", "std"),
    )
)

zero = pd.read_csv(ZERO)

assert len(zero) == 2400
assert zero["path"].nunique() == 80

merged = zero.merge(
    best,
    on="path",
    how="left",
    validate="many_to_one",
)

assert len(merged) == 2400
assert merged["bls_best20"].notna().all()
assert (merged["bls_best20"] > 0).all()

for decoder in ("05", "median", "rank50"):
    merged[f"ratio_{decoder}"] = (
        merged[f"cut{decoder}"] / merged["bls_best20"]
    )

summary = (
    merged.groupby(["model", "train_family", "test_family"])
    .agg(
        count=("path", "size"),
        cut_bls05=("ratio_05", "mean"),
        cut_blsmedian=("ratio_median", "mean"),
        cut_blsrank50=("ratio_rank50", "mean"),
    )
    .reset_index()
)

best.to_csv(OUT / "density_bls_best20.csv", index=False)
merged.to_csv(
    OUT / "density_ood_cut_bls_detailed.csv",
    index=False,
)
summary.to_csv(
    OUT / "density_ood_cut_bls_summary.csv",
    index=False,
)

print("\n=== DENSITY OOD CUT/BLS (%) ===")

display = (
    merged.groupby(["model", "test_family"])[
        ["ratio_05", "ratio_median", "ratio_rank50"]
    ].mean() * 100
)

print(display.round(2).to_string())

print("\n=== OVERALL (%) ===")
print(
    (merged.groupby("model")[
        ["ratio_05", "ratio_median", "ratio_rank50"]
    ].mean() * 100).round(2).to_string()
)

print("\nVALIDATION PASSED")
print("BLS runs:", len(bls))
print("Graphs:", len(best))
print("GNN evaluations:", len(merged))
