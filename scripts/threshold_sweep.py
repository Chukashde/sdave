import argparse
import csv
import json
import numpy as np
import torch

from scripts.output_diagnostic import select, load_rows
from scripts.seq_curriculum import make_ec, build_net, fwd
from scripts.g1_diag import make_x


def get_cut(G, p, threshold):
    b = p >= threshold
    return (
        G.w * (b[G.src] != b[G.dst])
    ).sum().item() / G.wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="dataset/splits/manifest.json")
    ap.add_argument("--output", default="results/threshold_sweep_100k.csv")
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

    thresholds = np.arange(0.1, 1.0, 0.1)
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

                fixed = get_cut(G, p, 0.5)
                median_t = torch.median(p).item()
                median_cut = get_cut(G, p, median_t)

                sweep = [
                    (float(t), get_cut(G, p, float(t)))
                    for t in thresholds
                ]

                best_t, best_cut = max(
                    sweep, key=lambda z: z[1]
                )

            results.append({
                "dataset": name,
                "path": path,
                "cut_fixed": fixed,
                "cut_median": median_cut,
                "cut_sweep": best_cut,
                "best_threshold": best_t,
                "gain": best_cut - fixed,
            })

        subset = [r for r in results if r["dataset"] == name]

        for col in ["cut_fixed", "cut_median", "cut_sweep", "gain"]:
            values = np.array([r[col] for r in subset]) * 100
            print(
                f"{col:12s}: "
                f"{values.mean():6.2f} ± {values.std():5.2f}",
                flush=True
            )

    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "path", "cut_fixed",
                "cut_median", "cut_sweep",
                "best_threshold", "gain"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSaved:", a.output)


if __name__ == "__main__":
    main()
