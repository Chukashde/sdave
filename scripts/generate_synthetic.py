"""Санамсаргүй синтетик граф (ER, SF, SBM) үүсгэж Gset форматаар хадгалдаг."""
import argparse
import os
import networkx as nx
import numpy as np


def save_gset(g: nx.Graph, path: str):
    g = nx.convert_node_labels_to_integers(g, first_label=0)
    n = g.number_of_nodes()
    edges = list(g.edges())
    with open(path, "w") as f:
        f.write(f"{n} {len(edges)}\n")
        for u, v in edges:
            f.write(f"{u+1} {v+1} 1\n")  # Gset 1-indexed


def make_er(n, p, seed):
    return nx.erdos_renyi_graph(n, p, seed=seed)


def make_sf(n, m, seed):
    # Barabasi-Albert: эхлээд чиглэлгүй, скейл-фри граф
    return nx.barabasi_albert_graph(n, m, seed=seed)


def make_sbm(n, k, p_in, p_out, seed):
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="node count")
    ap.add_argument("--count", type=int, default=5, help="graph бүрийн тоо (per type)")
    ap.add_argument("--out", default="dataset/synthetic")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    for i in range(args.count):
        seed = int(rng.randint(0, 1_000_000))

        # ER: дундаж degree ~10 болгох p сонгоно
        p = 10.0 / args.n
        g = make_er(args.n, p, seed)
        path = f"{args.out}/ER_n{args.n}_{i}.txt"
        save_gset(g, path)
        print(f"ER  [{i}] n={g.number_of_nodes()} e={g.number_of_edges()} -> {path}")

        # SF: Barabasi-Albert, m=5 (шинэ node холбогдох edge тоо)
        g = make_sf(args.n, 5, seed + 1)
        path = f"{args.out}/SF_n{args.n}_{i}.txt"
        save_gset(g, path)
        print(f"SF  [{i}] n={g.number_of_nodes()} e={g.number_of_edges()} -> {path}")

        # SBM: 10 блок, дотоод/гадаад магадлал
        g = make_sbm(args.n, 10, p_in=0.05, p_out=0.002, seed=seed + 2)
        path = f"{args.out}/SBM_n{args.n}_{i}.txt"
        save_gset(g, path)
        print(f"SBM [{i}] n={g.number_of_nodes()} e={g.number_of_edges()} -> {path}")


if __name__ == "__main__":
    main()
