"""Tests for the centralized config: default parity, YAML/CLI precedence."""
import torch

from pigat.config import PigatConfig, Config, load_config

# Frozen snapshot of the legacy Config defaults (excluding runtime device/dtype
# and the new first-class `penalty` field). Guards against default drift.
LEGACY_DEFAULTS = {
    "activation": "relu",
    "attn_drop": 0.1,
    "checkpoint_dir": "./checkpoints",
    "default_graph_name": "G22",
    "dim_embedding": 32,
    "dropout": 0.2,
    "feat_drop": 0.0,
    "fixed_hidden_dim_size": 16,
    "gradient_clip": None,
    "hidden_dim": [16],
    "hidden_layers_range": list(range(1, 21)),
    "learning_rate": 0.001,
    "log_interval": 100,
    "negative_slope": 0.2,
    "num_heads": 2,
    "number_classes": 1,
    "number_epochs": 10000,
    "output_dir": "./results",
    "patience": 10000,
    "prob_threshold": 0.5,
    "residual": False,
    "save_checkpoint": True,
    "seed": 42,
    "test_ratio": 0.1,
    "tolerance": 0.0001,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "weight_decay": 0.0,
}


def test_config_alias():
    assert Config is PigatConfig


def test_default_parity():
    d = PigatConfig().to_dict()
    for key, expected in LEGACY_DEFAULTS.items():
        assert d[key] == expected, f"default drift for {key}: {d[key]!r} != {expected!r}"


def test_runtime_dtype_is_torch():
    assert PigatConfig().to_dict()["dtype"] is torch.float32


def test_precedence_defaults_yaml_cli(tmp_path):
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text("num_heads: 1\nattn_drop: 0.0\ndropout: 0.3\n")
    # CLI override beats YAML; None override is ignored; YAML beats default.
    cfg = load_config(str(yaml_path), overrides={"num_heads": 4, "dropout": None})
    assert cfg.num_heads == 4      # CLI override
    assert cfg.attn_drop == 0.0    # from YAML
    assert cfg.dropout == 0.3      # YAML kept (None override ignored)
    assert cfg.patience == 10000   # default untouched


def test_yaml_roundtrip(tmp_path):
    cfg = load_config(overrides={"num_heads": 3})
    out = tmp_path / "rt.yaml"
    cfg.save_yaml(str(out))
    again = PigatConfig.from_yaml(str(out))
    assert again.to_dict() == cfg.to_dict()
    assert again.dtype is torch.float32
