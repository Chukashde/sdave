"""Centralized configuration for pigat experiments.

PigatConfig is a dataclass whose field defaults mirror the original Config
exactly. Configuration is YAML-primary: defaults are overlaid by an optional
YAML file and then by explicit (CLI) overrides. The effective config can be
serialized back to YAML and saved next to a results CSV for reproducibility.
"""
import copy as _copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from pigat.training.device import resolve_device

_DTYPES = {'float32': torch.float32, 'float64': torch.float64, 'float16': torch.float16}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}


@dataclass
class PigatConfig:
    """All experiment parameters. Defaults match the legacy Config."""

    # Training
    number_epochs: int = 10000
    learning_rate: float = 0.001
    prob_threshold: float = 0.5
    tolerance: float = 1e-4
    patience: int = 10000
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = None

    # Model architecture
    dim_embedding: int = 32
    hidden_dim: List[int] = field(default_factory=lambda: [16])
    dropout: float = 0.2
    number_classes: int = 1
    activation: str = 'relu'

    # GAT-specific
    num_heads: int = 2
    feat_drop: float = 0.0
    attn_drop: float = 0.1
    negative_slope: float = 0.2
    residual: bool = False

    # Experiment / sweep
    hidden_layers_range: List[int] = field(default_factory=lambda: list(range(1, 21)))
    fixed_hidden_dim_size: int = 16
    output_dir: str = './results'
    default_graph_name: str = 'G22'

    # Data split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Logging
    log_interval: int = 100
    save_checkpoint: bool = True
    checkpoint_dir: str = './checkpoints'

    # MIS-specific (set dynamically on the legacy Config; a first-class field here)
    penalty: int = 10

    # Runtime
    seed: int = 42
    device: Optional[str] = None
    dtype: Any = torch.float32

    def __post_init__(self):
        # Resolve device the same way the legacy Config did (cuda/cpu auto,
        # mps -> cpu). Allow dtype to be given as a string from YAML.
        self.device = resolve_device(self.device)
        if isinstance(self.dtype, str):
            self.dtype = _DTYPES[self.dtype]

    # --- dict / mutation API (compatible with the legacy Config) ---
    def to_dict(self) -> Dict[str, Any]:
        """Runtime dict consumed by the trainer/runner (dtype is a torch.dtype)."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    def update(self, **kwargs) -> None:
        for key, value in kwargs.items():
            if key == 'dtype' and isinstance(value, str):
                value = _DTYPES[value]
            setattr(self, key, value)

    def copy(self) -> 'PigatConfig':
        return _copy.deepcopy(self)

    # --- YAML I/O ---
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PigatConfig':
        cfg = cls()
        cfg.update(**data)
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> 'PigatConfig':
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def to_yaml_dict(self) -> Dict[str, Any]:
        """YAML-safe dict (dtype as a string name)."""
        data = self.to_dict()
        data['dtype'] = _DTYPE_NAMES.get(self.dtype, 'float32')
        return data

    def save_yaml(self, path: str) -> None:
        import yaml
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.to_yaml_dict(), f, sort_keys=True)


def load_config(yaml_path: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> PigatConfig:
    """Build a config with precedence: defaults < YAML file < explicit overrides.

    Args:
        yaml_path: optional path to a YAML config file.
        overrides: optional mapping of field -> value (e.g. parsed CLI args).
            Keys whose value is None are ignored so unset CLI flags do not
            clobber YAML values.
    """
    cfg = PigatConfig.from_yaml(yaml_path) if yaml_path else PigatConfig()
    if overrides:
        cfg.update(**{k: v for k, v in overrides.items() if v is not None})
    return cfg


def get_default_config() -> PigatConfig:
    """Return a default configuration instance (legacy helper)."""
    return PigatConfig()


# Backwards-compatible alias for the legacy class name.
Config = PigatConfig
