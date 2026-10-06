"""Splits + reference cut үүсгэнэ."""
import argparse, json, os, hashlib
import numpy as np, networkx as nx
from scripts.generate_synthetic import save_gset, make_er, make_sf, make_sbm
from scripts.evaluate import local_search
from scripts.train_multi import prep, cut_value

def make_rr(n, s):  return nx.random_regular_graph(10, n, seed=s)
def make_ws(n, s):  return nx.connected_watts_strogatz_graph(n, 10, 0.1, seed=s)
def make_geo(n, s): return nx.random_geometric_graph(n, (10 / (np.pi * n)) ** 0.5, seed=s)

def er(d): return lambda n, s: make_er(n, d / n, s)
FAM = {"ER": er(10), "SF": lambda n, s: make_sf(n, 5, s),
       "SBM": lambda n, s: make_sbm(n, 10, 50.0 / n, 2.0 / n, s),
       "RR": make_rr, "WS": make_ws, "GEO": make_geo,
       "ER_d4": er(4), "ER_d20": er(20)}

PLAN = [
 ("train", ["ER","SF","SBM"], 1000, 50, 1_000_000),
 ("val", ["ER","SF","SBM"], 1000, 5, 2_000_000),
 ("test_iid", ["ER","SF","SBM"], 1000, 10, 3_000_000),
 ("test_size", ["ER","SF","SBM"], 500, 10, 4_000_000),
 ("test_size", ["ER","SF","SBM"], 2000, 10, 5_000_000),
 ("test_size", ["ER","SF","SBM"], 5000, 5, 7_000_000),
 ("test_degree", ["ER_d4","ER_d20"], 1000, 10, 8_000_000),
 ("test_family", ["RR","WS","GEO"], 1000, 10, 6_000_000),
]

def ref_cut(path, restarts, seed=0):
    g, e, w = prep(path)
    n = g.number_of_nodes()
    adj = [[] for _ in range(n)]
    for (u, v), x in zip(e.T, w):
        adj[u].append((v, x)); adj[v].append((u, x))
    rng = np.random.RandomState(seed)
    best = max(cut_value(local_search(adj, rng.randint(0, 2, n)), e, w)
               for _ in range(restarts))
    return float(best), float(w.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset/splits")
    ap.add_argument("--restarts", type=int, default=20)
    a = ap.parse_args()

    manifest, seen = [], set()
    for split, fams, n, count, base in PLAN:
        d = os.path.join(a.out, split); os.makedirs(d, exist_ok=True)
        for fam in fams:
            for i in range(count):
                seed = base + i
                g = nx.Graph(FAM[fam](n, seed))
                g.remove_edges_from(nx.selfloop_edges(g))
                g = nx.convert_node_labels_to_integers(g)
                name = f"{fam}_n{n}_s{seed}"
                path = os.path.join(d, name + ".txt")
                save_gset(g, path)
                h = hashlib.md5(open(path, "rb").read()).hexdigest()
                assert h not in seen, f"duplicate: {name}"; seen.add(h)
                ref, tot = ref_cut(path, a.restarts)
                manifest.append(dict(split=split, family=fam, n=g.number_of_nodes(),
                    edges=g.number_of_edges(), seed=seed, path=path,
                    components=nx.number_connected_components(g),
                    ref_cut=ref, ref_ratio=ref / tot))
        print(split, n, fams, flush=True)

    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"Total {len(manifest)} graphs")

if __name__ == "__main__":
    main()
