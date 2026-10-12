import argparse
import csv
import json
import numpy as np
import torch
from collections import defaultdict

from scripts.output_diagnostic import select, load_rows
from scripts.seq_curriculum import make_ec, build_net, fwd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bls", default="results/bls_final.csv")
    ap.add_argument("--manifest", default="dataset/splits/manifest.json")
    ap.add_argument("--output", default="results/gnn_vs_bls_100k.csv")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    ec = make_ec(a)
    net = build_net(a, ec)

    ckpt = torch.load(
        a.checkpoint, map_location=ec["device"],
        weights_only=True
    )
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    net.load_state_dict(ckpt)
    net.eval()

    # BLS: graph path -> best cut across 5 seeds
    bls = defaultdict(list)
    with open(a.bls, newline="") as f:
        for r in csv.DictReader(f):
            bls[r["path"]].append(float(r["cut"]))

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

    output = []

    for name, rows in sets.items():
        items = load_rows(rows, ec, a.k)
        print(f"\n{name}: {len(items)} graphs", flush=True)

        for row, item in zip(rows, items):
            # Expected load_rows output: (G, U, ...)
            Gk, U = item["G"], item["U"]

            from scripts.g1_diag import make_x
            x = make_x(Gk, "clean", a.dim, U, None, ec)

            with torch.no_grad():
                p = fwd(net, Gk, x)
                b = (p >= 0.5)
                gnn_cut = (
                    Gk.w * (b[Gk.src] != b[Gk.dst])
                ).sum().item()

            path = row["path"]
            if path not in bls:
                print("Missing BLS:", path)
                continue

            best_bls = max(bls[path])
            ratio = gnn_cut / best_bls if best_bls else float("nan")

            output.append({
                "dataset": name,
                "path": path,
                "gnn_cut": gnn_cut,
                "bls_best_cut": best_bls,
                "gnn_bls_ratio": ratio,
            })

        subset = [r for r in output if r["dataset"] == name]
        if subset:
            vals = np.array([r["gnn_bls_ratio"] for r in subset])
            print(
                f"Ratio = {vals.mean():.4f} ± {vals.std():.4f}",
                flush=True
            )

    with open(a.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "path", "gnn_cut",
                "bls_best_cut", "gnn_bls_ratio"
            ]
        )
        writer.writeheader()
        writer.writerows(output)

    print("\nSaved:", a.output)


if __name__ == "__main__":
    main()
