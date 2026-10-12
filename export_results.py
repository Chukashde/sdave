import os
import re
import csv
import glob

LOGDIR = "logs"
OUTDIR = "results"
os.makedirs(OUTDIR, exist_ok=True)

def read(path):
    with open(path, "r", errors="ignore") as f:
        return f.read()

def write_csv(name, rows, fields):
    path = os.path.join(OUTDIR, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{name:30s} {len(rows):3d} rows")

# ============================================================
# 1. MULTI-GRAPH IID
# ============================================================

rows = []

for path in sorted(glob.glob(f"{LOGDIR}/multi_clean_seed*.log")):
    m = re.search(r"seed(\d+)", path)
    if not m:
        continue

    seed = int(m.group(1))
    text = read(path)

    loss = re.search(
        r"step 9999 loss_norm\(avg500\)\s+([-\d.]+)", text
    )

    train = re.search(
        r"train graphs.*?cut@0\.5\s+([\d.]+)%\s+cut@median\s+([\d.]+)%",
        text
    )

    unseen = re.search(
        r"UNSEEN:.*?cut@0\.5\s+([\d.]+)%\s+cut@median\s+([\d.]+)%",
        text
    )

    if train and unseen:
        tr = float(train.group(1))
        un = float(unseen.group(1))

        rows.append({
            "seed": seed,
            "steps": 10000,
            "final_loss": float(loss.group(1)) if loss else "",
            "train_cut05": tr,
            "train_median": float(train.group(2)),
            "unseen_cut05": un,
            "unseen_median": float(unseen.group(2)),
            "gap": round(un - tr, 3)
        })

write_csv(
    "multi_iid.csv",
    rows,
    [
        "seed", "steps", "final_loss",
        "train_cut05", "train_median",
        "unseen_cut05", "unseen_median", "gap"
    ]
)

# ============================================================
# 2. OOD TOPOLOGY
# ============================================================

rows = []

pattern = re.compile(
    r"^(TRAIN|IID SBM|OOD SIZE|OOD RR|OOD WS|OOD GEO)\s*"
    r"\|\s*cut@0\.5\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*median\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*n=(\d+)",
    re.MULTILINE
)

for path in sorted(glob.glob(f"{LOGDIR}/ood_seed*.log")):
    m = re.search(r"seed(\d+)", path)
    if not m:
        continue

    seed = int(m.group(1))
    text = read(path)

    for x in pattern.finditer(text):
        rows.append({
            "seed": seed,
            "split": x.group(1),
            "cut05_mean": float(x.group(2)),
            "cut05_graph_std": float(x.group(3)),
            "median_mean": float(x.group(4)),
            "median_graph_std": float(x.group(5)),
            "n_graphs": int(x.group(6))
        })

write_csv(
    "ood_topology.csv",
    rows,
    [
        "seed", "split",
        "cut05_mean", "cut05_graph_std",
        "median_mean", "median_graph_std",
        "n_graphs"
    ]
)

# ============================================================
# 3. SIZE GENERALIZATION
# ============================================================

rows = []

size_pattern = re.compile(
    r"^(IID n=1000|OOD n=500\s|OOD n=2000|OOD n=5000)\s*"
    r"\|\s*cut@0\.5\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*median\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*n=(\d+)",
    re.MULTILINE
)

for path in sorted(glob.glob(f"{LOGDIR}/size_seed*.log")):
    m = re.search(r"seed(\d+)", path)
    if not m:
        continue

    seed = int(m.group(1))
    text = read(path)

    for x in size_pattern.finditer(text):
        label = x.group(1).strip()

        n_match = re.search(r"n=(\d+)", label)
        graph_size = int(n_match.group(1)) if n_match else ""

        split = "IID" if label.startswith("IID") else "OOD"

        rows.append({
            "seed": seed,
            "split": split,
            "graph_size": graph_size,
            "cut05_mean": float(x.group(2)),
            "cut05_graph_std": float(x.group(3)),
            "median_mean": float(x.group(4)),
            "median_graph_std": float(x.group(5)),
            "n_graphs": int(x.group(6))
        })

write_csv(
    "size_generalization.csv",
    rows,
    [
        "seed", "split", "graph_size",
        "cut05_mean", "cut05_graph_std",
        "median_mean", "median_graph_std",
        "n_graphs"
    ]
)

# ============================================================
# 4. NORMALIZATION RAW / NODE / EDGE
# ============================================================

rows = []

norm_pattern = re.compile(
    r"^(TRAIN n=500|TRAIN n=1000|TRAIN n=2000|IID n=1000|OOD n=5000)\s*"
    r"\|\s*cut@0\.5\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*median\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*n=(\d+)",
    re.MULTILINE
)

for mode in ["raw", "node", "edge"]:
    for path in sorted(glob.glob(f"{LOGDIR}/norm_{mode}_seed*.log")):
        m = re.search(r"seed(\d+)", path)
        if not m:
            continue

        seed = int(m.group(1))
        text = read(path)

        loss_matches = re.findall(
            r"step\s+\d+\s+loss\(avg500\)\s+([-\d.]+)",
            text
        )

        final_loss = float(loss_matches[-1]) if loss_matches else ""

        for x in norm_pattern.finditer(text):
            label = x.group(1)

            if label.startswith("TRAIN"):
                split = "TRAIN"
            elif label.startswith("IID"):
                split = "IID"
            else:
                split = "OOD"

            size = int(re.search(r"n=(\d+)", label).group(1))

            rows.append({
                "mode": mode.upper(),
                "seed": seed,
                "split": split,
                "graph_size": size,
                "final_loss": final_loss,
                "cut05_mean": float(x.group(2)),
                "cut05_graph_std": float(x.group(3)),
                "median_mean": float(x.group(4)),
                "median_graph_std": float(x.group(5)),
                "n_graphs": int(x.group(6))
            })

write_csv(
    "normalization.csv",
    rows,
    [
        "mode", "seed", "split", "graph_size",
        "final_loss",
        "cut05_mean", "cut05_graph_std",
        "median_mean", "median_graph_std",
        "n_graphs"
    ]
)

# ============================================================
# 5. GRADIENT NORMALIZATION
# ============================================================

rows = []

path = f"{LOGDIR}/gradient_norm_seed1.log"

if os.path.exists(path):
    text = read(path)

    current_mode = None

    for line in text.splitlines():
        mm = re.search(r"LOSS MODE:\s*(RAW|NODE|EDGE)", line)

        if mm:
            current_mode = mm.group(1)
            continue

        mm = re.search(
            r"n=(\d+)\s*\|\s*"
            r"loss\s+([-\d.]+)\s*\+/-\s*([\d.]+)\s*\|\s*"
            r"grad_norm\s+([\d.]+)\s*\+/-\s*([\d.]+)",
            line
        )

        if mm and current_mode:
            rows.append({
                "seed": 1,
                "mode": current_mode,
                "graph_size": int(mm.group(1)),
                "loss_mean": float(mm.group(2)),
                "loss_std": float(mm.group(3)),
                "grad_norm_mean": float(mm.group(4)),
                "grad_norm_std": float(mm.group(5))
            })

write_csv(
    "gradient_normalization.csv",
    rows,
    [
        "seed", "mode", "graph_size",
        "loss_mean", "loss_std",
        "grad_norm_mean", "grad_norm_std"
    ]
)

# ============================================================
# 6. DIVERSITY
# ============================================================

def parse_diversity(pattern_glob, steps, output_name):

    rows = []

    for path in glob.glob(pattern_glob):

        filename = os.path.basename(path)

        if steps == 100000:
            m = re.match(
                r"diversity100k_n(\d+)_seed(\d+)\.log",
                filename
            )
        else:
            m = re.match(
                r"diversity_n(\d+)_seed(\d+)\.log",
                filename
            )

        if not m:
            continue

        ntrain = int(m.group(1))
        seed = int(m.group(2))

        text = read(path)

        step_number = steps - 1

        loss = re.search(
            rf"step {step_number} loss_norm\(avg500\)\s+([-\d.]+)",
            text
        )

        train = re.search(
            r"train graphs.*?cut@0\.5\s+([\d.]+)%\s+"
            r"cut@median\s+([\d.]+)%",
            text
        )

        unseen = re.search(
            r"UNSEEN:.*?cut@0\.5\s+([\d.]+)%\s+"
            r"cut@median\s+([\d.]+)%",
            text
        )

        if train and unseen:

            train_cut = float(train.group(1))
            unseen_cut = float(unseen.group(1))

            rows.append({
                "ntrain": ntrain,
                "seed": seed,
                "steps": steps,
                "final_loss": float(loss.group(1)) if loss else "",
                "train_cut05": train_cut,
                "train_median": float(train.group(2)),
                "unseen_cut05": unseen_cut,
                "unseen_median": float(unseen.group(2)),
                "gap": round(unseen_cut - train_cut, 3)
            })

    rows.sort(key=lambda x: (x["ntrain"], x["seed"]))

    write_csv(
        output_name,
        rows,
        [
            "ntrain", "seed", "steps", "final_loss",
            "train_cut05", "train_median",
            "unseen_cut05", "unseen_median",
            "gap"
        ]
    )

    return rows

div10 = parse_diversity(
    f"{LOGDIR}/diversity_n*_seed*.log",
    10000,
    "diversity_10k.csv"
)

div100 = parse_diversity(
    f"{LOGDIR}/diversity100k_n*_seed*.log",
    100000,
    "diversity_100k.csv"
)

# ============================================================
# 7. OOD BASELINES
# ============================================================

rows = []

path = f"{LOGDIR}/ood_baselines.log"

baseline_pattern = re.compile(
    r"^(IID SBM|OOD SIZE|OOD RR|OOD WS|OOD GEO)\s*"
    r"\|\s*Random\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*Spectral\s+([\d.]+)\s*\+/-\s*([\d.]+)%"
    r"\s*\|\s*n=(\d+)",
    re.MULTILINE
)

if os.path.exists(path):
    text = read(path)

    for x in baseline_pattern.finditer(text):
        rows.append({
            "split": x.group(1),
            "random_mean": float(x.group(2)),
            "random_std": float(x.group(3)),
            "spectral_mean": float(x.group(4)),
            "spectral_std": float(x.group(5)),
            "n_graphs": int(x.group(6))
        })

write_csv(
    "baselines.csv",
    rows,
    [
        "split",
        "random_mean", "random_std",
        "spectral_mean", "spectral_std",
        "n_graphs"
    ]
)

# ============================================================
# 8. SINGLE TRANSFER
# ============================================================

rows = []

path = f"{LOGDIR}/single_transfer.log"

if os.path.exists(path):
    text = read(path)

    # Table-like result:
    # graph | train/unseen | numeric columns...
    for line in text.splitlines():

        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]

        if len(parts) < 3:
            continue

        if parts[1] not in ("train", "unseen"):
            continue

        graph = parts[0]
        split = parts[1]

        values = []
        for p in parts[2:]:
            p = p.replace("%", "").strip()
            try:
                values.append(float(p))
            except ValueError:
                pass

        row = {
            "graph": graph,
            "split": split
        }

        for i, value in enumerate(values, start=1):
            row[f"metric_{i}"] = value

        rows.append(row)

max_metrics = max(
    [len([k for k in r if k.startswith("metric_")]) for r in rows],
    default=0
)

fields = ["graph", "split"] + [
    f"metric_{i}" for i in range(1, max_metrics + 1)
]

write_csv(
    "single_transfer.csv",
    rows,
    fields
)

# ============================================================
# 9. MASTER CSV
# ============================================================

master = []

for r in div10:
    master.append({
        "experiment": "training_diversity",
        "setting": "SBM_IID",
        "seed": r["seed"],
        "ntrain": r["ntrain"],
        "steps": r["steps"],
        "train_cut05": r["train_cut05"],
        "test_cut05": r["unseen_cut05"],
        "train_median": r["train_median"],
        "test_median": r["unseen_median"],
        "gap": r["gap"]
    })

for r in div100:
    master.append({
        "experiment": "training_diversity",
        "setting": "SBM_IID",
        "seed": r["seed"],
        "ntrain": r["ntrain"],
        "steps": r["steps"],
        "train_cut05": r["train_cut05"],
        "test_cut05": r["unseen_cut05"],
        "train_median": r["train_median"],
        "test_median": r["unseen_median"],
        "gap": r["gap"]
    })

write_csv(
    "all_experiments.csv",
    master,
    [
        "experiment", "setting", "seed",
        "ntrain", "steps",
        "train_cut05", "test_cut05",
        "train_median", "test_median",
        "gap"
    ]
)

print()
print("Done.")
print("CSV files saved in:", OUTDIR)
