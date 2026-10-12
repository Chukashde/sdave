import argparse
import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss
from scripts.g1_diag import spectral, make_x, cut


def select(man, family, split, n=None):
    rows = [
        r for r in man
        if r["family"] == family and r["split"] == split
    ]
    rows = sorted(rows, key=lambda r: r["seed"])

    if n is not None:
        rows = [r for r in rows if r.get("n") == n]

    return rows


def load_rows(rows, ec, k):
    out = []

    for r in rows:
        Gk = G(r["path"], ec)
        U, _ = spectral(Gk, k)

        out.append({
            "row": r,
            "G": Gk,
            "U": U
        })

    return out


def evaluate(name, items, net, a, ec):
    losses = []
    cut05 = []
    cutmed = []

    means = []
    stds = []
    mins = []
    maxs = []
    medians = []
    frac05 = []

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    net.eval()

    with torch.no_grad():
        for item in items:
            r = item["row"]
            Gk = item["G"]
            U = item["U"]

            x = make_x(
                Gk,
                "clean",
                a.dim,
                U,
                None,
                ec
            )

            p = fwd(net, Gk, x)

            loss = norm_loss(p, Gk).item()

            c05 = cut(
                Gk,
                (p >= 0.5).long()
            )

            cmed = cut(
                Gk,
                (p >= p.median()).long()
            )

            pm = p.mean().item()
            ps = p.std().item()
            pmin = p.min().item()
            pmax = p.max().item()
            pmedian = p.median().item()
            f05 = (p >= 0.5).float().mean().item()

            losses.append(loss)
            cut05.append(c05)
            cutmed.append(cmed)

            means.append(pm)
            stds.append(ps)
            mins.append(pmin)
            maxs.append(pmax)
            medians.append(pmedian)
            frac05.append(f05)

            graph_name = r["path"].split("/")[-1]

            print(
                f"{graph_name:35s} "
                f"loss={loss:8.4f} "
                f"cut05={c05*100:6.2f}% "
                f"cutmed={cmed*100:6.2f}% "
                f"mean={pm:7.3f} "
                f"std={ps:7.3f} "
                f"median={pmedian:7.3f} "
                f"min={pmin:7.3f} "
                f"max={pmax:7.3f} "
                f">=0.5={f05*100:6.2f}%"
            )

    print("-" * 80)

    print(
        f"SUMMARY {name:12s} | "
        f"loss {np.mean(losses):.4f} | "
        f"cut05 {np.mean(cut05)*100:.2f}% | "
        f"median {np.mean(cutmed)*100:.2f}% | "
        f"p.mean {np.mean(means):.3f} | "
        f"p.std {np.mean(stds):.3f} | "
        f"p.median {np.mean(medians):.3f} | "
        f"p.min {np.mean(mins):.3f} | "
        f"p.max {np.mean(maxs):.3f} | "
        f">=0.5 {np.mean(frac05)*100:.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--checkpoint", required=True)

    ap.add_argument(
        "--manifest",
        default="dataset/splits/manifest.json"
    )

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
        a.checkpoint,
        map_location=ec["device"]
    )

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        net.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"])
    else:
        net.load_state_dict(ckpt)

    man = json.load(open(a.manifest))

    sets = {
        "IID_SBM":
            select(man, "SBM", "test_iid"),

        "SIZE_500":
            select(man, "SBM", "test_size", 500),

        "SIZE_2000":
            select(man, "SBM", "test_size", 2000),

        "SIZE_5000":
            select(man, "SBM", "test_size", 5000),

        "OOD_RR":
            select(man, "RR", "test_family"),

        "OOD_WS":
            select(man, "WS", "test_family"),

        "OOD_GEO":
            select(man, "GEO", "test_family"),
    }

    for name, rows in sets.items():
        print(f"{name}: {len(rows)} graphs")

    for name, rows in sets.items():
        items = load_rows(rows, ec, a.k)
        evaluate(name, items, net, a, ec)


if __name__ == "__main__":
    main()

