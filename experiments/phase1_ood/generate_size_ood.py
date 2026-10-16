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
    ap.add_argument('--out', default='experiments/phase1_ood/data')
    ap.add_argument('--per-size', type=int, default=10)
    ap.add_argument('--sizes', type=int, nargs='+', default=[600, 1000, 2000])
    args = ap.parse_args()
    root = Path(args.out)
    manifest_path = root / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError(f'Refusing to overwrite {manifest_path}')
    rows = []
    seen = set()
    for family in ('ER', 'RRG'):
        for n in args.sizes:
            if n <= 500 or n % 2:
                raise ValueError('OOD sizes must be even and > 500')
            for i in range(args.per_size):
                seed = 8_000_000 + (100_000 if family == 'ER' else 0) + n * 100 + i
                if family == 'RRG':
                    graph = nx.random_regular_graph(3, n, seed=seed)
                else:
                    graph = nx.erdos_renyi_graph(n, 4.0 / (n - 1), seed=seed)
                graph = nx.convert_node_labels_to_integers(graph)
                graph.remove_edges_from(nx.selfloop_edges(graph))
                digest = graph_hash(graph)
                if digest in seen:
                    raise RuntimeError('Duplicate graph hash')
                seen.add(digest)
                path = root / family / f'{family}_n{n}_s{seed}.txt'
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite {path}')
                path.parent.mkdir(parents=True, exist_ok=True)
                save_gset(graph, str(path))
                degrees = [d for _, d in graph.degree()]
                rows.append(dict(split='test_size_ood', family=family, n=n,
                                 edges=graph.number_of_edges(), seed=seed,
                                 path=str(path), mean_degree=sum(degrees) / n,
                                 components=nx.number_connected_components(graph),
                                 clustering=nx.average_clustering(graph),
                                 graph_hash=digest))
                print(f'GENERATED {family} n={n} i={i+1}', flush=True)
    manifest_path.write_text(json.dumps(rows, indent=2) + '\n')
    print(f'SAVED {manifest_path}: {len(rows)} graphs')


if __name__ == '__main__':
    main()
