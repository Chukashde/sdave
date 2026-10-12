import argparse
import torch
import numpy as np

from scripts.seq_curriculum import G, make_ec, build_net, fwd, norm_loss
from scripts.g1_diag import spectral, make_x


def cut_ratio(Gk, bits):
    return (
        Gk.w * (bits[Gk.src] != bits[Gk.dst])
    ).sum().item() / Gk.wsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--graph", required=True)
    args = ap.parse_args()

    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    a = argparse.Namespace(**ckpt["args"])
    ec = make_ec(a)

    net = build_net(a, ec)
    net.load_state_dict(ckpt["model_state"])
    net.eval()

    Gk = G(args.graph, ec)
    U, _ = spectral(Gk, a.k)

    x = make_x(Gk, "clean", a.dim, U, None, ec)

    with torch.no_grad():
        p = fwd(net, Gk, x)

        b05 = (p >= 0.5)
        bmed = (p >= p.median())

        print("Model:", a.model)
        print("Family:", a.family)
        print("Nodes:", Gk.n)
        print("p.min:", p.min().item())
        print("p.max:", p.max().item())
        print("p.mean:", p.mean().item())
        print("p.std:", p.std().item())
        print("nodes >= 0.5:", int(b05.sum().item()))
        print("nodes >= median:", int(bmed.sum().item()))
        print("loss:", norm_loss(p, Gk).item())
        print("cut@0.5:", cut_ratio(Gk, b05))
        print("cut@median:", cut_ratio(Gk, bmed))


if __name__ == "__main__":
    main()
