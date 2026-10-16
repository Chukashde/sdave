import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
from scripts.generate_synthetic import save_gset


def graph_hash(graph):
    edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
    payload = json.dumps([graph.number_of_nodes(), edges], separators=(',', ':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='experiments/phase1_ood/data_density')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--per-degree', type=int, default=10)
    args = ap.parse_args()

    root = Path(args.out)
    manifest_path = root / 'manifest.json'

    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite {manifest_path}')

    n = args.n
    if n <= 500 or n % 2:
        raise ValueError('n must be even and > 500')

    rows = []
    seen = set()

    degrees = {
        'ER': [4, 8, 12, 20],
        'RRG': [3, 6, 10, 20],
    }

    for family, degree_values in degrees.items():
        for degree in degree_values:
            if family == 'RRG' and (degree >= n or (n * degree) % 2):
                raise ValueError('Invalid regular graph degree')

            for i in range(args.per_degree):
                seed = (
                    9_000_000
                    + (100_000 if family == 'ER' else 200_000)
                    + degree * 1000
                    + i
                )

                if family == 'ER':
                    graph = nx.erdos_renyi_graph(
                        n, degree / (n - 1), seed=seed
                    )
                else:
                    graph = nx.random_regular_graph(
                        degree, n, seed=seed
                    )

                graph = nx.convert_node_labels_to_integers(graph)
                graph.remove_edges_from(nx.selfloop_edges(graph))

                digest = graph_hash(graph)
                if digest in seen:
                    raise RuntimeError('Duplicate graph hash')
                seen.add(digest)

                path = (
                    root / family /
                    f'{family}_n{n}_d{degree}_s{seed}.txt'
                )

                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite {path}')

                path.parent.mkdir(parents=True, exist_ok=True)
                save_gset(graph, str(path))

                actual_degree = (
                    2 * graph.number_of_edges() / n
                )

                rows.append(dict(
                    split='test_degree_ood',
                    family=family,
                    n=n,
                    edges=graph.number_of_edges(),
                    seed=seed,
                    target_degree=degree,
                    mean_degree=actual_degree,
                    path=str(path),
                    components=nx.number_connected_components(graph),
                    clustering=nx.average_clustering(graph),
                    graph_hash=digest,
                ))

                print(
                    f'GENERATED {family} n={n} '
                    f'degree={degree} graph={i+1}',
                    flush=True
                )

    manifest_path.write_text(
        json.dumps(rows, indent=2) + '\n'
    )
    print(f'SAVED {manifest_path}: {len(rows)} graphs')


if __name__ == '__main__':
    main()
