"""Train/val/test split бүхий синтетик граф dataset үүсгэнэ (Gset формат)."""
import argparse, json, os, hashlib
import networkx as nx
from scripts.generate_synthetic import save_gset, make_er, make_sf, make_sbm


def make_rr(n, seed):   return nx.random_regular_graph(10, n, seed=seed)
def make_ws(n, seed):   return nx.connected_watts_strogatz_graph(n, 10, 0.1, seed=seed)
def make_geo(n, seed):  return nx.random_geometric_graph(n, (10 / (3.14159 * n)) ** 0.5, seed=seed)

FAMILIES = {
    "ER":  lambda n, s: make_er(n, 10.0 / n, s),
    "SF":  lambda n, s: make_sf(n, 5, s),
    "SBM": lambda n, s: make_sbm(n, 10, 50.0 / n, 2.0 / n, s),
    "RR":  make_rr,
    "WS":  make_ws,
    "GEO": make_geo,
}

# (split, families, n, count per family, seed_base)
PLAN = [
    ("train",       ["ER", "SF", "SBM"], 1000, 20, 1_000_000),
    ("val",         ["ER", "SF", "SBM"], 1000,  5, 2_000_000),
    ("test_iid",    ["ER", "SF", "SBM"], 1000, 10, 3_000_000),
    ("test_size",   ["ER", "SF", "SBM"],  500, 10, 4_000_000),
    ("test_size",   ["ER", "SF", "SBM"], 2000, 10, 5_000_000),
    ("test_family", ["RR", "WS", "GEO"], 1000, 10, 6_000_000),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/splits")
    args = ap.parse_args()

    manifest, seen = [], set()
    for split, fams, n, count, base in PLAN:
        d = os.path.join(args.out, split)
        os.makedirs(d, exist_ok=True)
        for fam in fams:
            for i in range(count):
                seed = base + i
                g = FAMILIES[fam](n, seed)
                g = nx.Graph(g)
                g.remove_edges_from(nx.selfloop_edges(g))
                name = f"{fam}_n{n}_s{seed}"
                path = os.path.join(d, name + ".txt")
                save_gset(g, path)
                h = hashlib.md5(open(path, "rb").read()).hexdigest()
                assert h not in seen, f"давхардсан граф: {name}"
                seen.add(h)
                manifest.append({
                    "split": split, "family": fam, "n": g.number_of_nodes(),
                    "edges": g.number_of_edges(), "seed": seed, "path": path,
                    "components": nx.number_connected_components(g),
                })
        print(f"{split}: n={n} {fams} x{count}")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Нийт {len(manifest)} граф -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()
