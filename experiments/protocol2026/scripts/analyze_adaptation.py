import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


BASE = Path("experiments/protocol2026/results/production")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def mean(values):
    return statistics.mean(values)


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    refs = {
        r["path"]: float(r["reference_cut"])
        for r in read_csv(BASE / "bls_test_reference_200.csv")
    }
    assert len(refs) == 200

    zero = read_csv(BASE / "zero_shot.csv")
    assert len(zero) == 6000

    zero_map = {}
    for r in zero:
        key = (
            r["model"], r["train_family"],
            int(r["train_seed"]), r["path"]
        )
        assert key not in zero_map
        assert r["path"] in refs
        zero_map[key] = float(r["cut05"]) / refs[r["path"]]

    adaptation_dir = BASE / "adaptation"
    files = sorted(
        p for p in adaptation_dir.glob("*.csv")
        if not p.name.startswith(".")
    )

    if not files:
        print("No completed adaptation checkpoints yet.")
        return

    all_rows = []
    expected_pairs = {
        (regime, step)
        for regime in ("fine_tune", "from_scratch")
        for step in (10, 100, 1000)
    }

    for path in files:
        rows = read_csv(path)
        groups = defaultdict(set)

        for r in rows:
            groups[r["path"]].add(
                (r["regime"], int(r["steps"]))
            )

        valid = (
            len(rows) == 1200
            and len(groups) == 200
            and all(v == expected_pairs for v in groups.values())
        )

        if not valid:
            raise RuntimeError(f"Incomplete checkpoint: {path}")

        all_rows.extend(rows)

    if len(files) != 30 and not args.allow_partial:
        print(
            f"Completed: {len(files)}/30. "
            "Waiting for all checkpoints."
        )
        return

    # One score per graph, seed, regime, step.
    per_graph = defaultdict(dict)

    for r in all_rows:
        model = r["model"]
        train_family = r["train_family"]
        test_family = r["test_family"]
        seed = int(r["train_seed"])
        path = r["path"]
        regime = r["regime"]
        step = int(r["steps"])

        key = (model, train_family, seed, path)
        assert key in zero_map
        assert path in refs
        assert test_family in ("ER", "RRG")
        assert abs(
            float(r["reference_cut"]) - refs[path]
        ) < 1e-9

        group = (model, train_family, test_family, seed)
        item = (regime, step, path)

        assert item not in per_graph[group]

        score = float(r["cut05"]) / refs[path]
        per_graph[group][item] = (
            score, float(r["adaptation_sec"])
        )

    seed_results = []

    for group, items in sorted(per_graph.items()):
        model, train_family, test_family, seed = group

        paths = {
            path for regime, step, path in items
        }

        if len(paths) != 100:
            raise RuntimeError(
                f"Expected 100 graphs: {group}, got {len(paths)}"
            )

        for regime in ("fine_tune", "from_scratch"):
            for step in (10, 100, 1000):
                scores = []
                gains = []
                times = []

                for path in paths:
                    score, elapsed = items[(regime, step, path)]
                    baseline = zero_map[
                        (model, train_family, seed, path)
                    ]

                    scores.append(score)
                    gains.append(score - baseline)
                    times.append(elapsed)

                seed_results.append({
                    "model": model,
                    "train_family": train_family,
                    "test_family": test_family,
                    "seed": seed,
                    "regime": regime,
                    "steps": step,
                    "mean_ratio": mean(scores),
                    "mean_delta_vs_zero": mean(gains),
                    "mean_adaptation_sec": mean(times),
                })

    output = BASE / "adaptation_seed_summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(seed_results[0])
        )
        writer.writeheader()
        writer.writerows(seed_results)

    groups = defaultdict(list)

    for r in seed_results:
        key = (
            r["model"], r["train_family"],
            r["test_family"], r["regime"], r["steps"]
        )
        groups[key].append(r)

    summary = []

    for key, rows in sorted(groups.items()):
        model, train_family, test_family, regime, step = key

        if len(rows) != 5 and not args.allow_partial:
            raise RuntimeError(f"Missing seeds: {key}")

        summary.append({
            "model": model,
            "train_family": train_family,
            "test_family": test_family,
            "regime": regime,
            "steps": step,
            "n_seeds": len(rows),
            "ratio_mean": mean([r["mean_ratio"] for r in rows]),
            "ratio_seed_sd": sd([r["mean_ratio"] for r in rows]),
            "delta_vs_zero_mean": mean([
                r["mean_delta_vs_zero"] for r in rows
            ]),
            "adaptation_sec_mean": mean([
                r["mean_adaptation_sec"] for r in rows
            ]),
        })

    output = BASE / "adaptation_summary.csv"
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Completed checkpoint files: {len(files)}/30")
    print(f"Adaptation rows: {len(all_rows)}")
    print(f"Seed summary: {len(seed_results)} rows")
    print(f"Final summary: {len(summary)} rows")
    print("ADAPTATION ANALYSIS PASSED")

    print("\nPI-GATv2-JK, 1000 steps:")
    for r in summary:
        if r["model"] == "pigatv2_jk" and r["steps"] == 1000:
            print(
                f'{r["train_family"]}->{r["test_family"]} '
                f'{r["regime"]}: '
                f'{r["ratio_mean"]:.4f} '
                f'+/- {r["ratio_seed_sd"]:.4f}'
            )


if __name__ == "__main__":
    main()
