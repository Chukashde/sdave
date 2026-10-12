
import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd
from scripts.g1_diag import spectral, make_x, cut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="dataset/splits/manifest.json")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", default="pigatv2")
    ap.add_argument("--config", default="configs/maxcut_main.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    ec = make_ec(a)
    net = build_net(a, ec)

    state = torch.load(
        a.checkpoint,
        map_location=ec["device"],
        weights_only=True
    )
    net.load_state_dict(state)
    net.eval()

    manifest = json.load(open(a.manifest))

    groups = [
        ("IID_SBM", "SBM", "test_iid"),
        ("OOD_GEO", "GEO", "test_family"),
        ("OOD_WS", "WS", "test_family"),
    ]

    with torch.no_grad():
        for name, family, split in groups:
            rows = sorted(
                [
                    r for r in manifest
                    if r["family"] == family
                    and r["split"] == split
                ],
                key=lambda r: r["seed"]
            )

            if not rows:
                print(f"{name}: no graphs found")
                continue

            # Эхлээд бүлэг бүрийн нэг graph дээр шалгана.
            row = rows[0]
            Gk = G(row["path"], ec)

            features = []
            predictions = []
            cuts = []

            print(f"\n===== {name} =====")
            print("Graph:", row["path"])

            for repeat in range(a.repeats):
                # ARPACK-ийн эхлэх векторыг өөрчилнө.
                # Graph болон checkpoint өөрчлөгдөхгүй.
                np.random.seed(a.seed + repeat)

                U, _ = spectral(Gk, a.k)
                x = make_x(
                    Gk, "clean", a.dim, U, None, ec
                )
                p = fwd(net, Gk, x)

                c = cut(Gk, (p >= 0.5).long()) * 100

                features.append(x.detach().cpu().numpy())
                predictions.append(p.detach().cpu().numpy())
                cuts.append(c)

                print(
                    f"repeat={repeat + 1:2d} "
                    f"cut={c:7.2f}% "
                    f"p.mean={p.mean().item():.4f} "
                    f"p.std={p.std().item():.4f}"
                )

            x0 = features[0]
            p0 = predictions[0]

            feature_drift = [
                np.linalg.norm(x - x0) /
                (np.linalg.norm(x0) + 1e-12)
                for x in features[1:]
            ]

            prediction_drift = [
                np.mean(np.abs(p - p0))
                for p in predictions[1:]
            ]

            print("--- SUMMARY ---")
            print(
                f"cut mean={np.mean(cuts):.2f}% "
                f"std={np.std(cuts):.2f}pp "
                f"min={np.min(cuts):.2f}% "
                f"max={np.max(cuts):.2f}%"
            )
            print(
                f"feature drift mean="
                f"{np.mean(feature_drift):.6f}"
            )
            print(
                f"prediction drift mean="
                f"{np.mean(prediction_drift):.6f}"
            )


if __name__ == "__main__":
    main()

