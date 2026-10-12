import itertools

import networkx as nx
import numpy as np
import torch

from pigat.eval.maxcut import (
    gen_q_dict_maxcut,
    qubo_dict_to_torch,
    loss_func,
    evaluate_maxcut_solution,
)


def main():
    rng = np.random.default_rng(42)
    total_assignments = 0

    for test_id in range(20):
        n = int(rng.integers(4, 13))
        probability = float(rng.uniform(0.2, 0.8))

        graph = nx.erdos_renyi_graph(
            n, probability, seed=test_id
        )

        for u, v in graph.edges():
            graph[u][v]["weight"] = int(rng.integers(1, 6))

        q_dict = gen_q_dict_maxcut(graph)
        Q = qubo_dict_to_torch(
            graph,
            q_dict,
            torch_dtype=torch.float64,
            torch_device="cpu",
        )

        for bits in itertools.product((0, 1), repeat=n):
            x = torch.tensor(bits, dtype=torch.float64)

            energy = loss_func(x, Q).item()
            cut = evaluate_maxcut_solution(
                x, graph
            )["cut_value"]

            assert np.isclose(energy, -cut), (
                f"FAILED graph={test_id}, "
                f"bits={bits}, energy={energy}, cut={cut}"
            )

            total_assignments += 1

        print(
            f"Graph {test_id + 1}/20 PASSED "
            f"| n={n} | edges={graph.number_of_edges()}"
        )

    print("\nALL REAL MAX-CUT QUBO TESTS PASSED")
    print("Total assignments checked:", total_assignments)


if __name__ == "__main__":
    main()
