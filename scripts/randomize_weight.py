import networkx as nx
import numpy as np

from pigat.data import read_gset

def randomize_weights_like_real(G,
                                alpha=0.4,
                                lognorm_mu=0.0,
                                lognorm_sigma=1.0,
                                w_min=1.0,
                                w_max=10.0,
                                seed=0):
    """
    G: NetworkX Graph (unweighted)
    alpha: degree correlation хүч (0 → degree-ээс үл хамаарах)
    lognorm_mu, lognorm_sigma: log-normal суурь тархалтын параметр
    w_min, w_max: эцсийн жингийн интервал
    """
    rng = np.random.default_rng(seed)

    # ----- 1. degree array -----
    degrees = dict(G.degree())
    # node labels are 1..n (Gset)
    
    edges = list(G.edges())
    m = len(edges)
    
    # ----- 2. base log-normal weights -----
    base = rng.lognormal(mean=lognorm_mu, sigma=lognorm_sigma, size=m)
    
    # ----- 3. degree-based factor -----
    deg_factors = []
    for (u, v) in edges:
        du = degrees[u]
        dv = degrees[v]
        factor = (du * dv) ** alpha
        deg_factors.append(factor)
    deg_factors = np.array(deg_factors, dtype=np.float64)
    
    # normalize so that mean factor = 1
    deg_factors /= deg_factors.mean()
    
    # ----- 4. combine base * factor -----
    weights = base * deg_factors
    
    # optional: truncate extreme outliers (for nicer range)
    # e.g. clip to 99th percentile
    hi = np.percentile(weights, 99)
    weights = np.clip(weights, 0, hi)
    
    # ----- 5. rescale to [w_min, w_max] -----
    w_lo, w_hi = weights.min(), weights.max()
    if w_hi > w_lo:
        weights = (weights - w_lo) / (w_hi - w_lo)
        weights = w_min + weights * (w_max - w_min)
    else:
        # all equal
        weights = np.full_like(weights, (w_min + w_max) / 2.0)
    
    # ----- 6. round to integer if you want discrete weights -----
    weights = np.round(weights).astype(int)
    
    # ----- 7. assign to graph -----
    for (w_val, (u, v)) in zip(weights, edges):
        G[u][v]["weight"] = int(w_val)
    
    return G

def save_weighted_gset(G, path):
    """
    G: NetworkX Graph, жин 'weight' attribute дээр
    path: гарах файл
    """
    nodes = sorted(G.nodes())
    mapping = {old: new for new, old in enumerate(nodes, start=1)}
    G2 = nx.relabel_nodes(G, mapping)

    N = G2.number_of_nodes()
    M = G2.number_of_edges()

    with open(path, "w") as f:
        f.write(f"{N} {M}\n")
        for u, v, data in G2.edges(data=True):
            w = data.get("weight", 1)
            f.write(f"{u} {v} {w}\n")



if __name__ == "__main__":
    G, _ = read_gset("dataset/Gset/G70")
    G_w = randomize_weights_like_real(G,
                                    alpha=0.4,
                                    lognorm_mu=0.0,
                                    lognorm_sigma=0.9,
                                    w_min=1,
                                    w_max=100,
                                    seed=42)

    # Жин шалгах
    some_edge = next(iter(G_w.edges()))
    print(some_edge, G_w[some_edge[0]][some_edge[1]]["weight"])
    # Хадгалж байна
    save_weighted_gset(G_w, "dataset/Gset/G70_weighted")