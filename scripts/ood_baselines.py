import json
import numpy as np
import torch

from scripts.seq_curriculum import G, make_ec
from scripts.g1_diag import spectral, cut


class Args:
    model = 'pigatv2'
    config = 'configs/maxcut_main.yaml'
    device = 'cuda'
    dim = 32
    layers = 3
    heads = 1
    k = 16
    seed = 1
    lr = 1e-3


a = Args()
ec = make_ec(a)

np.random.seed(1)
torch.manual_seed(1)

man = json.load(open('dataset/splits/manifest.json'))


def get_rows(family, split):
    return sorted(
        [
            r for r in man
            if r['family'] == family
            and r['split'] == split
        ],
        key=lambda r: r['seed']
    )


def evaluate(name, rows):
    random_scores = []
    spectral_scores = []

    for r in rows:
        Gk = G(r['path'], ec)

        # -----------------------
        # Random baseline
        # -----------------------
        y_random = torch.randint(
            0,
            2,
            (Gk.n,),
            device=ec['device']
        )

        random_scores.append(
            cut(Gk, y_random)
        )

        # -----------------------
        # Spectral baseline
        # -----------------------
        U, _ = spectral(Gk, 16)

        # First spectral vector
        v = U[:, 0]

        # Sign split
        y_spectral = torch.tensor(
            v >= 0,
            dtype=torch.long,
            device=ec['device']
        )

        spectral_scores.append(
            cut(Gk, y_spectral)
        )

    rmean = np.mean(random_scores) * 100
    rstd = np.std(random_scores) * 100

    smean = np.mean(spectral_scores) * 100
    sstd = np.std(spectral_scores) * 100

    print(
        f"{name:15s} | "
        f"Random {rmean:5.1f} +/- {rstd:4.1f}% | "
        f"Spectral {smean:5.1f} +/- {sstd:4.1f}% | "
        f"n={len(rows)}"
    )


tests = [
    (
        'IID SBM',
        get_rows('SBM', 'test_iid')
    ),
    (
        'OOD SIZE',
        get_rows('SBM', 'test_size')
    ),
    (
        'OOD RR',
        get_rows('RR', 'test_family')
    ),
    (
        'OOD WS',
        get_rows('WS', 'test_family')
    ),
    (
        'OOD GEO',
        get_rows('GEO', 'test_family')
    ),
]


print()
print("========================================")
print("OOD BASELINES")
print("========================================")

for name, rows in tests:
    evaluate(name, rows)
