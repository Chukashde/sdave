import pickle
import numpy as np
import networkx as nx
import torch

np.random.seed(1)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)


def create_graph(graph_type):
    num_nodes = np.random.randint(5000, 10000)

    if graph_type == "ER":
        p = np.random.randint(2, 25) * 0.0001
        return nx.generators.random_graphs.fast_gnp_random_graph(num_nodes, p=p, directed=True)

    if graph_type == "SF":
        alpha = np.random.randint(40, 60) * 0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        return nx.scale_free_graph(num_nodes, alpha=alpha, beta=beta, gamma=gamma)

    if graph_type == "GRP":
        s = np.random.randint(200, 1000)
        v = np.random.randint(200, 1000)
        p_in = np.random.randint(2, 25) * 0.0001
        p_out = np.random.randint(2, 25) * 0.0001
        g = nx.generators.gaussian_random_partition_graph(
            num_nodes, s=s, v=v, p_in=p_in, p_out=p_out, directed=True)
        assert nx.is_directed(g), "Not directed"
        return g


def build_adj(g_nx):
    """Dense хөрш матриц A[u,v] = u->v edge-ийн тоо (multi-edge хадгална, NetworKit-тэй ижил)."""
    n = g_nx.number_of_nodes()
    e = np.array(list(g_nx.edges()), dtype=np.int64)
    A = torch.zeros(n, n, dtype=torch.float32)
    src = torch.from_numpy(e[:, 0])
    dst = torch.from_numpy(e[:, 1])
    A.index_put_((src, dst), torch.ones(len(e)), accumulate=True)
    return A.to(device)


def bfs_batch(A, srcs):
    """srcs-ээс зэрэг BFS. dist (-1 = хүрэх боломжгүй), sigma (богино замын тоо), max түвшин."""
    n = A.shape[0]
    B = len(srcs)
    ar = torch.arange(B, device=A.device)
    sigma = torch.zeros(B, n, device=A.device)
    dist = torch.full((B, n), -1.0, device=A.device)
    sigma[ar, srcs] = 1.0
    dist[ar, srcs] = 0.0
    d = 0
    while True:
        f = torch.where(dist == d, sigma, torch.zeros_like(sigma))
        new = f @ A
        newmask = (dist < 0) & (new > 0)
        if not bool(newmask.any()):
            break
        dist = torch.where(newmask, torch.full_like(dist, d + 1), dist)
        sigma = torch.where(newmask, new, sigma)
        d += 1
    return dist, sigma, d


@torch.no_grad()
def centralities(A, batch=256):
    n = A.shape[0]
    bc = torch.zeros(n, device=A.device)
    close = torch.zeros(n, device=A.device)
    At = A.t()

    for start in range(0, n, batch):
        srcs = torch.arange(start, min(start + batch, n), device=A.device)
        B = len(srcs)
        ar = torch.arange(B, device=A.device)
        dist, sigma, maxd = bfs_batch(A, srcs)

        # ---- Closeness (generalized, normalized) ----
        reach = (dist >= 0).sum(1).float()                 # өөрийгөө оруулаад
        ssum = torch.where(dist > 0, dist, torch.zeros_like(dist)).sum(1)
        c = (reach - 1) ** 2 / ((n - 1) * ssum.clamp(min=1))
        c = torch.where(ssum > 0, c, torch.zeros_like(c))
        close[srcs] = c

        # ---- Betweenness (Brandes, backward accumulation) ----
        delta = torch.zeros_like(sigma)
        sig = sigma.clamp(min=1)
        for d in range(maxd, 0, -1):
            coef = torch.where(dist == d, (1 + delta) / sig, torch.zeros_like(delta))
            t = coef @ At
            delta = delta + torch.where(dist == d - 1, sigma * t, torch.zeros_like(delta))
        delta[ar, srcs] = 0.0
        bc += delta.sum(0)

    bc = bc / ((n - 1) * (n - 2))                          # directed, normalized
    return bc.cpu().numpy(), close.cpu().numpy()


num_of_graphs = 50
graph_types = ["ER", "SF", "GRP"]

for graph_type in graph_types:
    print("###################")
    print(f"Generating graph type : {graph_type}")
    list_bet_data, list_close_data = [], []

    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}", end="\r")
        g_nx = create_graph(graph_type)

        if nx.number_of_isolates(g_nx) > 0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)

        A = build_adj(g_nx)
        bet, clo = centralities(A)
        bet_dict = {j: float(bet[j]) for j in range(len(bet))}
        close_dict = {j: float(clo[j]) for j in range(len(clo))}
        list_bet_data.append([g_nx, bet_dict])
        list_close_data.append([g_nx, close_dict])
        del A
        if device == "mps":
            torch.mps.empty_cache()

    with open(f"./graphs/{graph_type}_data_bet.pickle", "wb") as f:
        pickle.dump(list_bet_data, f)
    with open(f"./graphs/{graph_type}_data_close.pickle", "wb") as f:
        pickle.dump(list_close_data, f)
    print("\nGraphs saved")

print("End.")
