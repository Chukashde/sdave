import itertools
import numpy as np
import networkx as nx


def cut_value(G, x):
    return sum(
        G[u][v].get("weight", 1.0) * (x[u] != x[v])
        for u, v in G.edges()
    )


def qubo_energy(G, x):
    quadratic = sum(
        2 * G[u][v].get("weight", 1.0) * x[u] * x[v]
        for u, v in G.edges()
    )

    linear = sum(
        sum(G[u][v].get("weight", 1.0) for v in G.neighbors(u))
        * x[u]
        for u in G.nodes()
    )

    return quadratic - linear


def main():
    rng = np.random.default_rng(42)

    for test_id in range(20):
        n = int(rng.integers(4, 13))
        p = float(rng.uniform(0.2, 0.8))

        G = nx.erdos_renyi_graph(n, p, seed=test_id)

        for u, v in G.edges():
            G[u][v]["weight"] = float(rng.integers(1, 6))

        for bits in itertools.product((0, 1), repeat=n):
            cut = cut_value(G, bits)
            energy = qubo_energy(G, bits)

            assert np.isclose(energy, -cut), (
                f"FAILED test={test_id}, n={n}, "
                f"cut={cut}, energy={energy}"
            )

        print(f"Test {test_id + 1}/20 PASSED | n={n}")

    print("\nAll Max-Cut QUBO identity tests PASSED")


if __name__ == "__main__":
    main()
