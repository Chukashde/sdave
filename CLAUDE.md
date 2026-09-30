# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`pigat` is a research codebase: physics-inspired GNNs (PI-GCN, PI-GATv2) for MAX-CUT and MIS, backing a paper in revision. The central experiments are **depth sweeps** that study oversmoothing (Dirichlet energy, MAD, cosine similarity) versus solution quality. It is an installable package (`pip install -e .`); there is no `sys.path` hacking.

**Top constraint when editing:** preserve exact experimental behavior. Do not change model logic, the MAX-CUT/MIS QUBO loss, hyperparameter defaults, or seeding unless explicitly asked. The verification harness lives in the session scratchpad; the portable guard is `tests/test_cutvalue.py` (same seed -> identical cut value on CPU).

## Commands

```bash
pip install -e .                 # or: pip install -e . --no-deps  (if DGL already present)
python -m pytest tests/ -q       # loader / eval / config / determinism tests
python -m pytest tests/test_loader.py -q   # a single test file

# Single experiment
python scripts/run_experiment.py --graph dataset/Gset/G14 --model pigcn --seed 1 --layers 16 16 16

# Full depth sweep (YAML config + CLI overrides)
python scripts/run_sweep.py --config configs/maxcut_main.yaml --graph dataset/Gset/G14 --models pigcn pigatv2 --seeds 1 2 3
python scripts/run_sweep.py --config configs/mis.yaml --graph dataset/Gset/G14 --problem mis

# Regenerate figures from existing CSVs (no rerun)
python scripts/make_figures.py --results_dir results --output_dir plots

# Classical MAX-CUT baseline (self-contained)
python scripts/bls.py dataset/Gset/G14 --time 5 --seed 1
```

DGL has no MPS backend; `--device mps` maps to CPU (with a warning). See README for per-platform DGL install (CUDA on Ubuntu vs CPU on macOS).

## Architecture

Pipeline: **PigatConfig -> sweeps.run_sweep -> Trainer -> model**, results written as one superset CSV plus a sidecar `<csv>.config.yaml`.

- **`pigat/config/config.py`** — `PigatConfig` dataclass (aliased as `Config`). Defaults mirror the original Config exactly (guarded by `tests/test_config.py`). `load_config(yaml, overrides)` layers precedence defaults < YAML < CLI (argparse `SUPPRESS`, so unset flags do not clobber YAML). `to_dict()` returns the runtime dict (with a real `torch.dtype`); `save_yaml()` persists a YAML-safe copy.
- **`pigat/data/loaders.py`** — `load_graph(path, force=None)` auto-detects format by header token count (1 token -> adjacency list `read_graph`; >=2 int tokens -> Gset edge list `read_gset`) and dispatches to the unchanged parsers.
- **`pigat/models/`** — `PIBaseGNN` (base.py) holds shared bookkeeping and the forward interface but **constructs no layers** (so weight-init RNG order is owned by each subclass). `PIGCN` uses a plain `OrderedDict` of `GraphConv` (a pre-existing quirk: those params are unregistered and untrained — only the embedding learns). `PIGATv2` uses an `nn.ModuleList` of `GATv2Conv` with `feat_drop` deliberately tied to the shared dropout. `create_model` is the factory.
- **`pigat/training/`** — `trainer.py` (learnable `nn.Embedding` is the model input, co-optimized via one Adam; loss is the QUBO `xᵀQx`), `seed.py` (`setup_environment`, called per cell; CUDA seeding intentionally left commented), `device.py` (`resolve_device`, mps->cpu).
- **`pigat/eval/`** — `maxcut.py` and `mis.py` build the QUBO and evaluate solutions (they keep separate QUBO/loss copies on purpose); `smoothness.py` has the oversmoothing metrics.
- **`pigat/sweeps/`** — `driver.py::run_sweep` is the single `models x seeds x architectures` loop; `_run_cell` preserves the exact operation order `setup_environment -> dgl.from_networkx -> build_q -> create_model -> embedding -> train` (RNG-critical). `problems.py` has `MaxCutProblem`/`MISProblem` strategies providing `build_q`, `evaluate`, and the problem-specific CSV columns.
- **`pigat/viz/`** — `ExperimentVisualizer` reads result CSVs only (decoupled from sweeps) and relabels PIGATv2 -> "PIGAT" in figures.

## Conventions and gotchas

- The `num_heads`/`attn_drop` entry-point drift is intentional and explicit: `configs/maxcut_main.yaml` (1 head, 0.0) reproduces old `main.py` runs; `configs/maxcut_batch.yaml` (2 heads, 0.1) reproduces the old batch scripts. Do not silently unify them.
- One superset CSV schema reuses the historical column names so old result CSVs remain a subset and `ExperimentVisualizer` keeps parsing them. Do not rename columns the viz fuzzy-matchers key on (`cut_value`, `final_dirichlet_energy`, `final_cosine_sim`, `training_time`, ...).
- Never touch `results/`, `plots/`, `embed.pt`, `*.tar.gz`, `dataset/` — these are tracked experiment artifacts.
- Verification is CPU-only (DGL CUDA kernels are nondeterministic). Same seed on one machine -> identical cut values; cross-machine bit-equality is not expected.
