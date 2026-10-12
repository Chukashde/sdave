import argparse
import csv
import json
import numpy as np
import torch

from scripts.output_diagnostic import select, load_rows
from scripts.seq_curriculum import make_ec, build_net, fwd
from scripts.g1_diag import make_x


def exact_best_cut(G, p):
    """Find the best threshold cut in O((n+m) log n) time."""
    scores = p.detach().cpu().numpy()
    src = G.src.detach().cpu().numpy()
    dst = G.dst.detach().cpu().numpy()
    weights = G.w.detach().cpu().numpy()

    n = len(scores)
    order = np.argsort(-scores, kind="stable")

    # Adjacency lists
    adj = [[] for _ in range(n)]
    for u, v, w in zip(src, dst, weights):
        adj[int(u)].append((int(v), float(w)))
        adj[int(v)].append((int(u), float(w)))

    selected = np.zeros(n, dtype=bool)
    current = 0.0
    best = 0.0
    best_k = 0

    i = 0
    while i < n:
        j = i + 1

        # Equal scores must be moved together.
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1

        for idx in range(i, j):
            u = int(order[idx])

            for v, w in adj[u]:
                if selected[v]:
                    current -= w
                else:
                    current += w

            selected[u] = True

        if current > best:
            best = current
            best_k = j

        i = j

    return best / G.wsum, best_k / n


def cut_at(G, p, threshold):
    b = p >= threshold
    return (
        G.w * (b[G.src] != b[G.dst])
    ).sum().item() / G.wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="dataset/splits/manifest.json")
    ap.add_argument("--output", default="results/exact_threshold_100k.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    ec = make_ec(a)
    net = build_net(a, ec)

    ckpt = torch.load(
        a.checkpoint,
        map_location=ec["device"],
        weights_only=True
    )

    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    net.load_state_dict(ckpt)
    net.eval()

    man = json.load(open(a.manifest))

    sets = {
        "IID_SBM": select(man, "SBM", "test_iid"),
        "SIZE_500": select(man, "SBM", "test_size", 500),
        "SIZE_2000": select(man, "SBM", "test_size", 2000),
        "SIZE_5000": select(man, "SBM", "test_size", 5000),
        "OOD_RR": select(man, "RR", "test_family"),
        "OOD_WS": select(man, "WS", "test_family"),
        "OOD_GEO": select(man, "GEO", "test_family"),
    }

    results = []

    for name, rows in sets.items():
        print(f"\n{name}: {len(rows)} graphs", flush=True)
        items = load_rows(rows, ec, a.k)

        for item in items:
            G = item["G"]
            U = item["U"]
            path = item["row"]["path"]

            x = make_x(G, "clean", a.dim, U, None, ec)

            with torch.no_grad():
                p = fwd(net, G, x)
                fixed = cut_at(G, p, 0.5)
                median = cut_at(G, p, torch.median(p).item())

            exact, selected_fraction = exact_best_cut(G, p)

            if exact + 1e-6 < max(fixed, median):
                raise RuntimeError(
                    f"Exact cut smaller than fixed/median: {path}"
                )

            results.append({
                "dataset": name,
                "path": path,
                "fixed": fixed,
                "median": median,
                "exact": exact,
                "gain": exact - fixed,
                "selected_fraction": selected_fraction
            })

        subset = [r for r in results if r["dataset"] == name]

        for col in ["fixed", "median", "exact", "gain"]:
            v = np.array([r[col] for r in subset]) * 100
            print(
                f"{col:8s}: {v.mean():6.2f} ± {v.std():5.2f}",
                flush=True
            )

    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "path", "fixed", "median",
                "exact", "gain", "selected_fraction"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", a.output)


if __name__ == "__main__":
    main()
