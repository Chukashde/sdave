# pigat

Physics-inspired graph neural networks (PI-GCN and PI-GATv2) for combinatorial
optimization on graphs: Maximum Cut (MAX-CUT) and Maximum Independent Set (MIS).
The codebase is built for depth-sweep experiments that study oversmoothing
(Dirichlet energy, MAD, cosine similarity) alongside solution quality.

## Layout

```
pigat/                 importable package
  config/              PigatConfig dataclass + YAML/CLI loading
  data/                load_graph (auto-detects Gset vs adjacency-list)
  models/              base.py (PIBaseGNN) + pigcn.py + pigatv2.py
  training/            trainer.py, seed.py, device.py
  eval/                maxcut.py, mis.py, smoothness.py
  sweeps/              driver.py (one sweep loop) + problems.py
  viz/                 ExperimentVisualizer (figures from result CSVs)
scripts/               run_experiment.py, run_sweep.py, make_figures.py, bls.py
configs/               example YAML configs (maxcut_main, maxcut_batch, mis)
tests/                 loader / eval / config / determinism tests
dataset/               Gset, real_world, SBM graphs
results/, plots/       experiment outputs and figures (preserved)
```

The two models share `PIBaseGNN`. In figures, PIGATv2 is labelled "PIGAT".

## Installation

DGL is the one tricky dependency: it ships from its own wheel index and the
build differs per machine, so it is installed separately from the rest. Do not
pin a single torch+DGL+CUDA combination; install the build that matches the
machine, then install this package without dependencies.

### Ubuntu + NVIDIA GPU (CUDA build of DGL)

```bash
python -m venv .venv && source .venv/bin/activate
# Install the torch build matching your CUDA toolkit (example: CUDA 12.1)
pip install torch --index-url https://download.pytorch.org/whl/cu121
# Install the matching CUDA build of DGL from the DGL wheel index
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
# Install pigat and the remaining deps
pip install -e .
```

### MacBook (Apple M-series): CPU build of DGL

DGL has no Metal/MPS backend, so on macOS the graph and its message-passing
kernels run on CPU. PyTorch may use MPS for dense ops, but the GNN layers
(GraphConv / GATv2Conv) are DGL kernels and stay on CPU. Develop and smoke-test
on the Mac; run full sweeps on the CUDA machine.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                       # CPU/MPS build from PyPI
pip install dgl -f https://data.dgl.ai/wheels/repo.html   # CPU build
pip install -e .
```

`--device mps` is accepted everywhere and transparently maps to CPU for DGL
(with a warning). If DGL is already provisioned in your environment, install
pigat without touching it: `pip install -e . --no-deps`.

Optional test extra: `pip install -e ".[test]"`.

## Running experiments

Configuration is YAML-primary; any field can be overridden on the command line.
Example YAMLs in `configs/` encode the historical attention-head settings:
`maxcut_main.yaml` (1 head, attn_drop 0.0), `maxcut_batch.yaml` (2 heads,
attn_drop 0.1), `mis.yaml`.

### A single experiment

```bash
python scripts/run_experiment.py \
    --graph dataset/Gset/G14 --model pigcn --seed 1 --layers 16 16 16 \
    --epochs 10000 --device cpu
```

### A full depth sweep (K = 1..20 layers, multiple seeds)

```bash
# MAX-CUT, main.py-style configuration
python scripts/run_sweep.py --config configs/maxcut_main.yaml \
    --graph dataset/Gset/G14 --models pigcn pigatv2 --seeds 1 2 3 4 5

# MAX-CUT, batch-style configuration (2 heads, attn_drop 0.1)
python scripts/run_sweep.py --config configs/maxcut_batch.yaml \
    --graph dataset/Gset/G22

# MIS sweep
python scripts/run_sweep.py --config configs/mis.yaml \
    --graph dataset/Gset/G14 --problem mis
```

Each run writes a results CSV plus a `<csv>.config.yaml` capturing the exact
effective config, so any run can be reproduced. MAX-CUT and MIS share one
superset CSV schema (MAX-CUT rows carry `cut_value`; MIS rows carry the
independent-set metrics).

### Regenerating figures (no experiments rerun)

Plotting reads result CSVs only and is fully decoupled from running sweeps:

```bash
python scripts/make_figures.py --results_dir results --output_dir plots
python scripts/make_figures.py --file results/<some>_sweep_<timestamp>.csv --analyze_only
```

### Classical baseline

`scripts/bls.py` is a self-contained Breakout Local Search MAX-CUT solver for
comparison:

```bash
python scripts/bls.py dataset/Gset/G14 --time 5 --seed 1
```

## Reproducibility notes

- Seeding is centralized in `pigat.training.seed.setup_environment` and applied
  per experiment cell, so cut values are deterministic run-to-run on a given
  machine.
- DGL CUDA kernels and cross-machine BLAS differences mean GPU and CPU runs are
  not bit-identical; pin "the paper numbers come from machine X". CPU runs on
  one machine are reproducible (see `tests/test_cutvalue.py`).

## Tests

```bash
python -m pytest tests/ -q
```

Covers the graph loader, cut-value / MIS evaluation, config defaults and
YAML/CLI precedence, and run-to-run determinism (same seed -> identical cut
value and identical model weights).
