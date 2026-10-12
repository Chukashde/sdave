import yaml


def load_protocol(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_production_config(cfg):
    if cfg["freeze"]["approved"] is not True:
        raise ValueError("Phase 0 protocol is not approved")

    training = cfg["training"]

    required = [
        "max_steps",
        "early_stopping_patience",
        "weight_decay",
    ]

    for key in required:
        if training.get(key) is None:
            raise ValueError(f"Missing training.{key}")

    if training["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")

    if training["early_stopping_patience"] <= 0:
        raise ValueError("early_stopping_patience must be positive")

    if training["weight_decay"] < 0:
        raise ValueError("weight_decay cannot be negative")

    return True


def apply_training_config(args, cfg):
    """Apply approved protocol training settings to CLI arguments."""
    validate_production_config(cfg)

    training = cfg["training"]
    experiment = cfg["experiment_Ia"]
    dataset = cfg["dataset"]

    if args.model not in experiment["architectures"]:
        raise ValueError("Model is not approved by protocol")

    if args.seed not in experiment["seeds"]:
        raise ValueError("Seed is not approved by protocol")

    if args.family not in dataset["families"]:
        raise ValueError("Graph family is not approved by protocol")

    mapping = {
        "loss": experiment["loss"],
        "steps": training["max_steps"],
        "patience": training["early_stopping_patience"],
        "log_every": training["log_every"],
        "weight_decay": training["weight_decay"],
        "lr": training["learning_rate"],
        "dim": training["dim"],
        "layers": training["layers"],
        "heads": training["heads"],
        "k": training["spectral_k"],
        "feature_mode": training["feature_mode"],
        "train_limit": dataset["train_per_family"],
        "val_limit": dataset["validation_per_family"],
        "manifest": dataset["manifest"],
        "config": training["config"],
    }

    for key, value in mapping.items():
        setattr(args, key, value)

    return args
