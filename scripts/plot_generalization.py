import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt


LOG_DIR = "logs"
OUT_DIR = "results/figures"

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def save_fig(name):
    png = os.path.join(OUT_DIR, name + ".png")
    pdf = os.path.join(OUT_DIR, name + ".pdf")

    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def extract_value(text, label):
    """
    Example:
    IID SBM | cut@0.5 62.7 +/- 1.4% | median 62.2 +/- 1.6%
    """

    pattern = (
        re.escape(label)
        + r".*?cut@0\.5\s+([0-9.]+)"
    )

    m = re.search(pattern, text)

    if not m:
        return None

    return float(m.group(1))


def extract_size_value(text, n):
    """
    Example:
    OOD n=500 | cut@0.5 61.87 +/- 0.93%
    """

    pattern = (
        rf"OOD n={n}\s*"
        + r".*?cut@0\.5\s+([0-9.]+)"
    )

    m = re.search(pattern, text)

    if not m:
        return None

    return float(m.group(1))


# ============================================================
# 1. TOPOLOGY OOD
# ============================================================

def plot_topology():

    files = sorted(
        glob.glob(os.path.join(LOG_DIR, "ood_seed*.log"))
    )

    if not files:
        print("WARNING: no ood_seed*.log files")
        return

    labels = [
        "IID SBM",
        "OOD RR",
        "OOD WS",
        "OOD GEO"
    ]

    display = [
        "IID SBM",
        "RR",
        "WS",
        "GEO"
    ]

    values = {x: [] for x in labels}

    for path in files:

        text = open(
            path,
            "r",
            errors="ignore"
        ).read()

        for label in labels:

            v = extract_value(text, label)

            if v is not None:
                values[label].append(v)

    print("\nTOPOLOGY DATA")

    means = []
    stds = []

    for label in labels:

        arr = np.array(
            values[label],
            dtype=float
        )

        if len(arr) == 0:
            print(f"{label}: NOT FOUND")
            means.append(np.nan)
            stds.append(0)
            continue

        mean = arr.mean()

        std = (
            arr.std(ddof=1)
            if len(arr) > 1
            else 0
        )

        means.append(mean)
        stds.append(std)

        print(
            f"{label:10s}: "
            f"{mean:.2f} +/- {std:.2f} "
            f"(seeds={len(arr)})"
        )

    # Existing baseline experiment
    random_baseline = [
        50.6,
        49.9,
        50.0,
        50.1
    ]

    spectral_baseline = [
        72.7,
        70.1,
        63.1,
        55.7
    ]

    x = np.arange(len(display))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        x - width,
        means,
        width,
        yerr=stds,
        capsize=5,
        label="PI-GNN"
    )

    ax.bar(
        x,
        random_baseline,
        width,
        label="Random"
    )

    ax.bar(
        x + width,
        spectral_baseline,
        width,
        label="Spectral"
    )

    ax.set_ylabel("Cut ratio (%)")
    ax.set_xlabel("Test graph distribution")

    ax.set_title(
        "PI-GNN Topology Generalization"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(display)

    ax.set_ylim(0, 80)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_fig("topology_ood")


# ============================================================
# 2. SIZE GENERALIZATION
# ============================================================

def plot_size():

    files = sorted(
        glob.glob(os.path.join(LOG_DIR, "size_seed*.log"))
    )

    if not files:
        print("WARNING: no size_seed*.log files")
        return

    sizes = [
        500,
        1000,
        2000,
        5000
    ]

    values = {
        n: [] for n in sizes
    }

    for path in files:

        text = open(
            path,
            "r",
            errors="ignore"
        ).read()

        # IID n=1000
        v = extract_value(
            text,
            "IID n=1000"
        )

        if v is not None:
            values[1000].append(v)

        # OOD sizes
        for n in [500, 2000, 5000]:

            v = extract_size_value(
                text,
                n
            )

            if v is not None:
                values[n].append(v)

    means = []
    stds = []

    print("\nSIZE DATA")

    for n in sizes:

        arr = np.array(
            values[n],
            dtype=float
        )

        if len(arr) == 0:

            print(
                f"n={n}: NOT FOUND"
            )

            means.append(np.nan)
            stds.append(0)

            continue

        mean = arr.mean()

        std = (
            arr.std(ddof=1)
            if len(arr) > 1
            else 0
        )

        means.append(mean)
        stds.append(std)

        print(
            f"n={n:<5d}: "
            f"{mean:.2f} +/- {std:.2f} "
            f"(seeds={len(arr)})"
        )

    fig, ax = plt.subplots(
        figsize=(8, 5.5)
    )

    ax.errorbar(
        sizes,
        means,
        yerr=stds,
        marker="o",
        linewidth=2,
        capsize=5,
        label="PI-GNN"
    )

    # Random baseline
    ax.axhline(
        50,
        linestyle="--",
        linewidth=1.5,
        label="Random baseline"
    )

    # Mark training graph size
    ax.axvline(
        1000,
        linestyle=":",
        linewidth=1.5,
        label="Training size (n=1000)"
    )

    ax.set_xlabel(
        "Number of nodes"
    )

    ax.set_ylabel(
        "Cut ratio (%)"
    )

    ax.set_title(
        "PI-GNN Graph-Size Generalization"
    )

    ax.set_xticks(sizes)

    ax.grid(
        alpha=0.25
    )

    ax.legend()

    save_fig(
        "size_generalization"
    )


# ============================================================
# 3. LARGE GRAPH SEED ROBUSTNESS
# ============================================================

def plot_seed_robustness():

    files = sorted(
        glob.glob(
            os.path.join(
                LOG_DIR,
                "size_seed*.log"
            )
        )
    )

    seeds = []
    vals = []

    for path in files:

        text = open(
            path,
            "r",
            errors="ignore"
        ).read()

        m = re.search(
            r"size_seed(\d+)\.log",
            path
        )

        if not m:
            continue

        seed = int(
            m.group(1)
        )

        v = extract_size_value(
            text,
            5000
        )

        if v is not None:

            seeds.append(seed)
            vals.append(v)

    if not vals:
        print(
            "WARNING: n=5000 values not found"
        )
        return

    order = np.argsort(seeds)

    seeds = np.array(seeds)[order]
    vals = np.array(vals)[order]

    print("\nN=5000 SEEDS")

    for s, v in zip(
        seeds,
        vals
    ):
        print(
            f"seed {s}: {v:.2f}%"
        )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        [str(x) for x in seeds],
        vals
    )

    ax.axhline(
        50,
        linestyle="--",
        linewidth=1.5,
        label="Random baseline"
    )

    ax.set_xlabel(
        "Training seed"
    )

    ax.set_ylabel(
        "Cut ratio (%)"
    )

    ax.set_title(
        "Robustness on Large Graphs (n=5000)"
    )

    ax.set_ylim(
        0,
        max(70, vals.max() + 5)
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.legend()

    save_fig(
        "size_seed_robustness"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "Reading experiment logs..."
    )

    plot_topology()
    plot_size()
    plot_seed_robustness()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(
        f"Figures saved to: {OUT_DIR}/"
    )
